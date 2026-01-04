"""
SVI and SSVI volatility surface fitting routines.
"""

import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize
from scipy.stats import norm
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SVIParams:
    """Container for SVI parameters."""
    a: float
    b: float
    rho: float
    m: float
    sigma: float


@dataclass
class SSVIParams:
    """Container for SSVI parameters."""
    rho: float
    eta: float
    gamma: float  # Also called 'p' in some implementations


class SVIFitter:
    """
    Fits SVI (Stochastic Volatility Inspired) model to volatility smiles.
    """
    
    @staticmethod
    def svi_variance(k: np.ndarray, T: float, a: float, b: float, rho: float,
                    m: float, sigma: float) -> np.ndarray:
        """Calculate SVI total variance."""
        return 0.5 * (a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))) * T
    
    @staticmethod
    def svi_vol(k: np.ndarray, T: float, a: float, b: float, rho: float,
               m: float, sigma: float) -> np.ndarray:
        """Calculate SVI implied volatility."""
        total_variance = SVIFitter.svi_variance(k, T, a, b, rho, m, sigma)
        total_variance = np.maximum(1e-9, total_variance)
        return np.sqrt(total_variance / T)
    
    @staticmethod
    def _svi_objective(params: np.ndarray, k: np.ndarray, T: float,
                      target_vol: np.ndarray) -> np.ndarray:
        """Objective function for SVI fitting."""
        a, b, rho, m, sigma = params
        
        # Constraint penalties
        if sigma <= 0 or b < 0 or abs(rho) > 1:
            return np.inf * np.ones_like(k)
        
        model_vol = SVIFitter.svi_vol(k, T, a, b, rho, m, sigma)
        return (model_vol - target_vol)**2
    
    def fit(self, df_expiry: pd.DataFrame) -> Tuple[Optional[SVIParams], Optional[np.ndarray]]:
        """
        Fit SVI model to a single expiry slice.
        
        Args:
            df_expiry: DataFrame with options for one expiry
            
        Returns:
            Tuple of (SVIParams, fitted_vols) or (None, None) if fit fails
        """
        if df_expiry.empty:
            return None, None
        
        k = df_expiry["k"].values
        T = float(df_expiry["T"].iloc[0])
        target_vol = df_expiry["iv_clean"].values
        
        # Initial guess
        a0 = np.min(target_vol)**2 * T * 2
        b0 = 0.1
        rho0 = -0.5
        m0 = 0.0
        sigma0 = 0.1
        
        initial_guess = [a0, b0, rho0, m0, sigma0]
        bounds = ([-np.inf, 0, -1, -np.inf, 1e-3], 
                 [np.inf, np.inf, 1, np.inf, np.inf])
        
        try:
            result = least_squares(
                SVIFitter._svi_objective,
                initial_guess,
                bounds=bounds,
                args=(k, T, target_vol),
                method='trf',
                max_nfev=1000
            )
            
            if result.success:
                params = SVIParams(*result.x)
                fitted_vols = SVIFitter.svi_vol(k, T, *result.x)
                return params, fitted_vols
            else:
                logger.warning(f"SVI fitting failed for T={T:.3f}y: {result.message}")
                return None, None
        except Exception as e:
            logger.warning(f"Error during SVI fitting for T={T:.3f}y: {e}")
            return None, None
    
    @staticmethod
    def svi_w(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
        """SVI total variance function w(k)."""
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
    
    @staticmethod
    def svi_wprime(k: np.ndarray, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
        """First derivative of SVI w(k)."""
        return b * (rho + (k - m) / np.sqrt((k - m)**2 + sigma**2))
    
    @staticmethod
    def svi_wpp(k: np.ndarray, b: float, m: float, sigma: float) -> np.ndarray:
        """Second derivative of SVI w(k)."""
        return b * (sigma**2) / (((k - m)**2 + sigma**2)**1.5)
    
    @staticmethod
    def atm_metrics_from_params(params: SVIParams, T: float) -> Tuple[float, float, float]:
        """
        Calculate ATM IV, skew, and curvature from SVI parameters.
        
        Returns:
            Tuple of (ATM_IV, ATM_skew, ATM_curvature)
        """
        k0 = 0.0
        w0 = SVIFitter.svi_w(k0, params.a, params.b, params.rho, params.m, params.sigma)
        wp = SVIFitter.svi_wprime(k0, params.b, params.rho, params.m, params.sigma)
        wpp = SVIFitter.svi_wpp(k0, params.b, params.m, params.sigma)
        
        iv_atm = np.sqrt(max(w0, 0.0) / max(T, 1e-12))
        skew_atm = 0.5 * wp / np.sqrt(max(T * w0, 1e-12))
        curv_atm = 0.5 / np.sqrt(max(T, 1e-12)) * (
            - (wp**2) / (4 * (w0**1.5 + 1e-18)) + (wpp / (2 * np.sqrt(w0 + 1e-18)))
        )
        
        return float(iv_atm), float(skew_atm), float(curv_atm)


class SSVIFitter:
    """
    Fits SSVI (Surface SVI) model to entire volatility surface.
    """
    
    EPS_T = 1.0 / 365.0  # Minimum time to expiry
    EPS_SIG = 1e-12
    
    @staticmethod
    def phi(theta: np.ndarray, eta: float, gamma: float) -> np.ndarray:
        """SSVI phi function."""
        return eta * (theta ** (-gamma))
    
    @staticmethod
    def w_ssvi(k: np.ndarray, theta: np.ndarray, rho: float, eta: float, gamma: float) -> np.ndarray:
        """SSVI total variance function."""
        ph = SSVIFitter.phi(theta, eta, gamma)
        root = np.sqrt((ph * k + rho)**2 + (1.0 - rho**2))
        return 0.5 * theta * (1.0 + rho * ph * k + root)
    
    @staticmethod
    def _global_ssvi_obj(params: np.ndarray, k_vec: np.ndarray, w_mkt_vec: np.ndarray,
                        theta_vec: np.ndarray, wgt_vec: np.ndarray) -> float:
        """Objective function for global SSVI fitting."""
        rho, eta, gamma = params
        
        w_model = SSVIFitter.w_ssvi(k_vec, theta_vec, rho, eta, gamma)
        err = np.mean(wgt_vec * (w_model - w_mkt_vec)**2)
        
        # Constraint penalties
        pen = 0.0
        
        if abs(rho) >= 1.0:
            pen += 1e6 * (abs(rho) - 0.999)**2
        if gamma <= 0.0:
            pen += 1e6 * (abs(gamma) + 0.01)**2
        if gamma > 0.5:
            pen += 1e6 * (gamma - 0.5)**2
        if eta <= 0:
            pen += 1e6 * (abs(eta) + 0.01)**2
        
        # Gatheral/Jacquier no-arbitrage condition
        gj_val = eta * (1.0 + abs(rho))
        if gj_val > 2.0:
            pen += 1000.0 * (gj_val - 2.0)**2
        
        return err + pen
    
    def fit_global(self, df: pd.DataFrame, thetas: Dict[pd.Timestamp, float],
                  forwards: Dict[pd.Timestamp, float], asof: pd.Timestamp,
                  cap_percentile: float = 90.0) -> Dict[pd.Timestamp, Tuple[SSVIParams, float]]:
        """
        Fit global SSVI parameters to entire surface.
        
        Args:
            df: Cleaned options DataFrame
            thetas: Dictionary mapping expiry to ATM total variance
            forwards: Dictionary mapping expiry to forward price
            asof: As-of date
            cap_percentile: Percentile for capping weights
            
        Returns:
            Dictionary mapping expiry to (SSVIParams, fit_cost)
        """
        # Prepare data - handle column name variations
        t_col = 'T' if 'T' in df.columns else 't'
        iv_col = 'mid_iv' if 'mid_iv' in df.columns else 'iv_clean'
        expiry_col = 'expiry' if 'expiry' in df.columns else 'expiry_dt'
        
        df_fit = df[df[t_col] > self.EPS_T].copy()
        df_fit['theta'] = df_fit[expiry_col].map(thetas)
        df_fit['w_mkt'] = (df_fit[iv_col]**2) * df_fit[t_col]
        df_fit['wgt'] = self._build_weights(df_fit, cap_percentile)
        
        # Handle k_fwd column (may be k or k_fwd)
        k_col = 'k_fwd' if 'k_fwd' in df_fit.columns else 'k'
        if k_col not in df_fit.columns:
            # Calculate k_fwd if not present
            strike_col = 'K' if 'K' in df_fit.columns else 'strike'
            df_fit['F'] = df_fit[expiry_col].map(forwards)
            df_fit['k_fwd'] = np.log(df_fit[strike_col] / df_fit['F'])
            k_col = 'k_fwd'
        
        all_k = df_fit[k_col].values.astype(float)
        all_w = df_fit['w_mkt'].values.astype(float)
        all_theta = df_fit['theta'].values.astype(float)
        all_weights = df_fit['wgt'].values.astype(float)
        
        # Global optimization
        x0 = [-0.4, 0.5, 0.4]
        bounds = [(-0.999, 0.999), (0.001, 5.0), (0.001, 0.5)]
        
        logger.info("Running global SSVI optimization...")
        res = minimize(
            SSVIFitter._global_ssvi_obj,
            x0,
            args=(all_k, all_w, all_theta, all_weights),
            method='L-BFGS-B',
            bounds=bounds
        )
        
        rho_opt, eta_opt, gamma_opt = res.x
        final_cost = res.fun
        
        logger.info(f"Global SSVI fit: rho={rho_opt:.5f}, eta={eta_opt:.5f}, "
                   f"gamma={gamma_opt:.5f}, cost={final_cost:.6f}")
        
        # Create params for each expiry
        global_params = SSVIParams(float(rho_opt), float(eta_opt), float(gamma_opt))
        slice_params = {}
        
        for exp in thetas.keys():
            slice_params[exp] = (global_params, float(final_cost))
        
        return slice_params
    
    def _build_weights(self, df: pd.DataFrame, cap_percentile: float) -> np.ndarray:
        """Build sample weights based on spread and liquidity."""
        spread = df.get('spread', pd.Series(1.0, index=df.index))
        spread = spread.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=1e-6)
        w = 1.0 / spread
        
        cap = np.nanpercentile(w, cap_percentile)
        w = np.minimum(w, cap)
        
        # Liquidity bumps
        if 'oi' in df.columns:
            w *= (1.0 + 0.5 * df['oi'].rank(pct=True).fillna(0.0))
        if 'volume' in df.columns:
            w *= (1.0 + 0.25 * df['volume'].rank(pct=True).fillna(0.0))
        
        # Down-weight extreme |k|
        if 'k_fwd' in df.columns:
            k_abs = df['k_fwd'].abs()
        elif 'k' in df.columns:
            k_abs = df['k'].abs()
        else:
            k_abs = pd.Series(0.0, index=df.index)
        
        k_cut = np.nanpercentile(k_abs, 99.0)
        w *= (1.0 - 0.5 * (k_abs > k_cut))
        
        return w.values
    
    def estimate_theta_atm(self, expiry_slice: pd.DataFrame, F: float) -> float:
        """Estimate ATM total variance theta for an expiry."""
        # Handle different column name variations
        strike_col = 'K' if 'K' in expiry_slice.columns else 'strike'
        iv_col = 'mid_iv' if 'mid_iv' in expiry_slice.columns else 'iv_clean'
        t_col = 'T' if 'T' in expiry_slice.columns else 't'
        
        idx = (expiry_slice[strike_col] - F).abs().idxmin()
        sigma_atm = float(expiry_slice.loc[idx, iv_col])
        T = float(expiry_slice.loc[idx, t_col])
        return max(sigma_atm, self.EPS_SIG)**2 * max(T, self.EPS_SIG)
    
    def enforce_monotone_theta(self, thetas: Dict[pd.Timestamp, float],
                               asof: pd.Timestamp) -> Dict[pd.Timestamp, float]:
        """Enforce monotonicity in theta(t) to reduce calendar arbitrage."""
        exps_sorted = sorted(thetas.keys(), key=lambda e: (e - asof).days)
        theta_arr = np.array([thetas[e] for e in exps_sorted])
        theta_mono = np.maximum.accumulate(theta_arr)
        return {e: th for e, th in zip(exps_sorted, theta_mono)}

