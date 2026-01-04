"""
Feature engineering module for extracting RND (Risk-Neutral Density) moments.
"""

import logging
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.integrate import simpson

logger = logging.getLogger(__name__)


class RNDFeatureExtractor:
    """
    Extracts risk-neutral density features (skew, kurtosis, etc.) from volatility surfaces.
    """
    
    K_MIN_MULT = 0.1
    K_MAX_MULT = 4.0
    N_POINTS = 5000
    
    @staticmethod
    def ssvi_power_law(k: np.ndarray, theta: float, rho: float, eta: float, gamma: float) -> np.ndarray:
        """Calculate total implied variance w(k) using Power-Law SSVI."""
        phi = eta / (theta ** gamma)
        z = phi * k
        w = (theta / 2) * (1 + rho * z + np.sqrt((z + rho)**2 + (1 - rho**2)))
        return w
    
    @staticmethod
    def bs_call_price(S: float, K: np.ndarray, T: float, r: float, sigma: np.ndarray) -> np.ndarray:
        """Vectorized Black-Scholes Call Price."""
        sigma = np.maximum(sigma, 1e-6)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return price
    
    def process_expiry_slice(self, row: pd.Series, S0: float, r0: float) -> Optional[Dict]:
        """
        Process a single expiry slice to extract RND moments.
        
        Args:
            row: Series with SSVI parameters for one expiry
            S0: Current spot price
            r0: Risk-free rate
            
        Returns:
            Dictionary with RND features or None if processing fails
        """
        T = row['T']
        if T < 0.005:  # Skip very short expiries
            return None
        
        # Setup grid
        F = S0 * np.exp(r0 * T)
        K_grid = np.linspace(S0 * self.K_MIN_MULT, S0 * self.K_MAX_MULT, self.N_POINTS)
        h = K_grid[1] - K_grid[0]
        
        # Reconstruct SSVI volatility
        log_moneyness = np.log(K_grid / F)
        
        theta = row['theta']
        rho = row['ssvi_rho']
        eta = row['ssvi_eta']
        gamma = row['ssvi_p']  # Note: 'p' in params file maps to 'gamma'
        
        # Calculate variance w(k)
        w_grid = self.ssvi_power_law(log_moneyness, theta, rho, eta, gamma)
        sigma_grid = np.sqrt(np.maximum(1e-9, w_grid) / T)
        
        # Calculate call prices
        calls = self.bs_call_price(S0, K_grid, T, r0, sigma_grid)
        
        # Breeden-Litzenberger: f(K) = e^(rT) * d2C/dK2
        d2C = (calls[2:] - 2 * calls[1:-1] + calls[:-2]) / (h**2)
        f_grid = np.exp(r0 * T) * d2C
        
        K_pdf = K_grid[1:-1]
        f_grid = np.maximum(0, f_grid)  # Remove negative densities
        
        # Normalize
        total_prob = simpson(y=f_grid, x=K_pdf)
        if total_prob < 0.95:
            logger.warning(f"Grid too narrow for T={T:.3f}, total_prob={total_prob:.3f}")
            return None
        
        pdf = f_grid / total_prob
        
        # Compute moments
        mean_rn = simpson(y=K_pdf * pdf, x=K_pdf)
        martingale_error = abs(mean_rn - F) / F
        
        var_rn = simpson(y=((K_pdf - mean_rn)**2) * pdf, x=K_pdf)
        vol_rn = np.sqrt(var_rn)
        
        skew_rn = simpson(y=((K_pdf - mean_rn)**3) * pdf, x=K_pdf) / (vol_rn**3)
        kurt_rn = simpson(y=((K_pdf - mean_rn)**4) * pdf, x=K_pdf) / (vol_rn**4)
        
        return {
            'expiry': row['expiry'],
            'T': T,
            'F': F,
            'rnd_mean': mean_rn,
            'martingale_error': martingale_error,
            'rnd_vol': np.sqrt(simpson(y=((K_pdf / F - 1.0)**2) * pdf, x=K_pdf) / T),
            'rnd_skew': skew_rn,
            'rnd_kurtosis': kurt_rn,
        }
    
    def extract_features(self, ssvi_params_df: pd.DataFrame, S0: float, r0: float) -> pd.DataFrame:
        """
        Extract RND features for all expiries.
        
        Args:
            ssvi_params_df: DataFrame with SSVI parameters per expiry
            S0: Current spot price
            r0: Risk-free rate
            
        Returns:
            DataFrame with RND features
        """
        rnd_results = []
        
        for idx, row in ssvi_params_df.iterrows():
            res = self.process_expiry_slice(row, S0, r0)
            if res:
                rnd_results.append(res)
        
        if not rnd_results:
            logger.warning("No RND features extracted")
            return pd.DataFrame()
        
        rnd_df = pd.DataFrame(rnd_results)
        logger.info(f"Extracted RND features for {len(rnd_df)} expiries")
        return rnd_df



