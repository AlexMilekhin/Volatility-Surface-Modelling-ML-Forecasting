"""
Main pipeline script for volatility surface modeling and forecasting.
Can be run from command line: python main.py --ticker JPM
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime

from src.data_loader import OptionsDataLoader
from src.models.bs_engine import ImpliedVolatilityCalculator
from src.models.svi_fit import SVIFitter, SSVIFitter
from src.features import RNDFeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VolatilitySurfacePipeline:
    """
    Main pipeline for volatility surface modeling.
    """
    
    def __init__(self, ticker: str, data_dir: str = "data", metrics_dir: str = "metrics",
                 rate: float = 0.045):
        """
        Initialize pipeline.
        
        Args:
            ticker: Stock ticker symbol
            data_dir: Directory for data files
            metrics_dir: Directory for metrics output
            rate: Risk-free rate
        """
        self.ticker = ticker
        self.data_dir = Path(data_dir)
        self.metrics_dir = Path(metrics_dir)
        self.rate = rate
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_loader = OptionsDataLoader(ticker)
        self.iv_calculator = ImpliedVolatilityCalculator(ticker=ticker)
        self.svi_fitter = SVIFitter()
        self.ssvi_fitter = SSVIFitter()
        self.rnd_extractor = RNDFeatureExtractor()
    
    def run_full_pipeline(self) -> None:
        """Run the complete pipeline from data collection to forecasting."""
        logger.info(f"Starting pipeline for {self.ticker}")
        
        try:
            # Step 1: Data collection
            logger.info("Step 1: Fetching options data...")
            options_df = self.data_loader.fetch_options_chain()
            hist_df = self.data_loader.fetch_historical_prices()
            
            # Save raw data
            options_path, hist_path = self.data_loader.save_data(
                options_df, hist_df, str(self.data_dir)
            )
            logger.info(f"Saved data to {options_path}")
            
            # Step 2: IV computation
            logger.info("Step 2: Computing implied volatilities...")
            clean_df = self.iv_calculator.recompute_clean_iv_surface(options_df, r=self.rate)
            
            clean_path = self.data_dir / "cleaned_options_data.parquet"
            clean_df.to_parquet(clean_path, index=False)
            logger.info(f"Saved cleaned data to {clean_path}")
            
            # Step 3: SVI fitting
            logger.info("Step 3: Fitting SVI models...")
            svi_results = self._fit_svi_surface(clean_df)
            
            if svi_results:
                svi_metrics_path = self.metrics_dir / f"svi_metrics_{datetime.now().date()}.parquet"
                svi_results.to_parquet(svi_metrics_path, index=False)
                logger.info(f"Saved SVI metrics to {svi_metrics_path}")
            
            # Step 4: SSVI fitting
            logger.info("Step 4: Fitting SSVI surface...")
            ssvi_results = self._fit_ssvi_surface(clean_df)
            
            if ssvi_results:
                ssvi_metrics_path = self.metrics_dir / f"ssvi_metrics_{datetime.now().date()}.parquet"
                ssvi_results.to_parquet(ssvi_metrics_path, index=False)
                logger.info(f"Saved SSVI metrics to {ssvi_metrics_path}")
            
            # Step 5: RND feature extraction
            if ssvi_results is not None and not ssvi_results.empty:
                logger.info("Step 5: Extracting RND features...")
                rnd_features = self._extract_rnd_features(ssvi_results, hist_df)
                
                if rnd_features is not None and not rnd_features.empty:
                    rnd_path = self.metrics_dir / f"rnd_features_{datetime.now().date()}.parquet"
                    rnd_features.to_parquet(rnd_path, index=False)
                    logger.info(f"Saved RND features to {rnd_path}")
            
            logger.info("Pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
    
    def _fit_svi_surface(self, clean_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Fit SVI models to each expiry."""
        rows = []
        asof = clean_df['asof'].iloc[0] if 'asof' in clean_df.columns else datetime.now()
        
        for expiry, df_exp in clean_df.groupby("expiry"):
            params, fitted_vols = self.svi_fitter.fit(df_exp)
            
            if params is not None:
                T = float(df_exp["T"].iloc[0])
                iv_atm, skew_atm, curv_atm = SVIFitter.atm_metrics_from_params(params, T)
                
                rows.append({
                    'asof': asof,
                    'expiry': expiry,
                    'T': T,
                    'ATM_IV': iv_atm,
                    'ATM_skew': skew_atm,
                    'ATM_curvature': curv_atm,
                    'a': params.a,
                    'b': params.b,
                    'rho': params.rho,
                    'm': params.m,
                    'sigma': params.sigma,
                })
        
        if not rows:
            logger.warning("No SVI fits succeeded")
            return None
        
        return pd.DataFrame(rows).sort_values(['asof', 'T']).reset_index(drop=True)
    
    def _fit_ssvi_surface(self, clean_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Fit SSVI surface globally."""
        try:
            # Prepare data
            clean_df = clean_df.copy()
            clean_df['expiry'] = pd.to_datetime(clean_df['expiry'])
            
            # Calculate forwards and thetas
            forwards = {}
            thetas = {}
            asof = clean_df['asof'].iloc[0] if 'asof' in clean_df.columns else datetime.now()
            
            for exp, g in clean_df.groupby('expiry'):
                F = self.iv_calculator.infer_forward_from_parity(g, self.rate)
                if np.isfinite(F):
                    forwards[exp] = F
                    theta = self.ssvi_fitter.estimate_theta_atm(g, F)
                    thetas[exp] = theta
            
            if not thetas:
                logger.warning("No valid forwards/thetas for SSVI fitting")
                return None
            
            # Enforce monotone theta
            thetas = self.ssvi_fitter.enforce_monotone_theta(thetas, asof)
            
            # Prepare k_fwd column
            clean_df['F'] = clean_df['expiry'].map(forwards)
            strike_col = 'strike' if 'strike' in clean_df.columns else 'K'
            clean_df['k_fwd'] = np.log(clean_df[strike_col] / clean_df['F'])
            
            # Ensure mid_iv column exists (use iv_clean if not)
            if 'mid_iv' not in clean_df.columns and 'iv_clean' in clean_df.columns:
                clean_df['mid_iv'] = clean_df['iv_clean']
            
            # Fit global SSVI
            slice_params = self.ssvi_fitter.fit_global(
                clean_df, thetas, forwards, asof
            )
            
            # Create results DataFrame
            rows = []
            for exp, (params, cost) in slice_params.items():
                rows.append({
                    'asof': asof,
                    'expiry': exp,
                    'T': float(clean_df[clean_df['expiry'] == exp]['T'].iloc[0]),
                    'theta': thetas[exp],
                    'ssvi_rho': params.rho,
                    'ssvi_eta': params.eta,
                    'ssvi_p': params.gamma,
                    'fit_cost': cost,
                })
            
            return pd.DataFrame(rows).sort_values('T').reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"SSVI fitting failed: {e}", exc_info=True)
            return None
    
    def _extract_rnd_features(self, ssvi_params_df: pd.DataFrame,
                              hist_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract RND features from SSVI parameters."""
        try:
            # Get spot price
            if isinstance(hist_df.index, pd.DatetimeIndex):
                latest_date = hist_df.index.max()
                S0 = float(hist_df.loc[latest_date, 'Close'])
            else:
                S0 = float(hist_df['Close'].iloc[-1])
            
            # Get risk-free rate (simplified - could use actual rate)
            r0 = self.rate
            
            # Extract features
            rnd_df = self.rnd_extractor.extract_features(ssvi_params_df, S0, r0)
            return rnd_df
            
        except Exception as e:
            logger.error(f"RND feature extraction failed: {e}", exc_info=True)
            return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Volatility Surface Modeling Pipeline')
    parser.add_argument('--ticker', type=str, default='JPM', help='Stock ticker symbol')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--metrics-dir', type=str, default='metrics', help='Metrics directory')
    parser.add_argument('--rate', type=float, default=0.045, help='Risk-free rate')
    
    args = parser.parse_args()
    
    pipeline = VolatilitySurfacePipeline(
        ticker=args.ticker,
        data_dir=args.data_dir,
        metrics_dir=args.metrics_dir,
        rate=args.rate
    )
    
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()