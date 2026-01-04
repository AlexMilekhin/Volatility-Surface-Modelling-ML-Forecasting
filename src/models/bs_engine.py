"""
Black-Scholes pricing engine and implied volatility calculation.
"""

import logging
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from math import sqrt, exp, log
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class BlackScholesEngine:
    """
    Black-Scholes option pricing engine with vectorized operations.
    """
    
    @staticmethod
    def _d1(S: np.ndarray, K: np.ndarray, r: float, q: float, 
            sigma: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Calculate d1 parameter for Black-Scholes."""
        # Avoid division by zero and invalid log
        ratio = S / K
        # Only clip if ratio is non-positive (shouldn't happen with valid inputs, but safety check)
        ratio = np.where(ratio > 0, ratio, 1e-10)
        sqrt_T = np.sqrt(T)
        sigma_sqrt_T = sigma * sqrt_T
        # Avoid division by zero - use a very small epsilon instead of zero
        sigma_sqrt_T = np.maximum(sigma_sqrt_T, 1e-10)
        return (np.log(ratio) + (r - q + 0.5 * sigma**2) * T) / sigma_sqrt_T
    
    @staticmethod
    def _d2(d1: np.ndarray, sigma: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Calculate d2 parameter for Black-Scholes."""
        return d1 - sigma * np.sqrt(T)
    
    @staticmethod
    def price(is_call: np.ndarray, S: np.ndarray, K: np.ndarray, r: float, 
              q: float, sigma: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Calculate Black-Scholes option prices (vectorized).
        
        Args:
            is_call: Boolean array indicating call (True) or put (False)
            S: Spot prices
            K: Strike prices
            r: Risk-free rate
            q: Dividend yield
            sigma: Volatilities
            T: Time to expiration (years)
            
        Returns:
            Array of option prices
        """
        # Handle edge cases
        # For very small sigma, return intrinsic value directly to avoid numerical precision issues
        # This ensures consistency with intrinsic calculation in implied_vol
        SIGMA_THRESHOLD = 1e-5
        mask_very_small_sigma = sigma < SIGMA_THRESHOLD
        mask_valid = (sigma > 0) & (T > 0) & (S > 0) & (K > 0)
        prices = np.zeros_like(S, dtype=float)
        
        # Calculate intrinsic value (used for very small sigma and invalid cases)
        intrinsic_call = np.maximum(0.0, S * np.exp(-q * T) - K * np.exp(-r * T))
        intrinsic_put = np.maximum(0.0, K * np.exp(-r * T) - S * np.exp(-q * T))
        
        if not mask_valid.any():
            # Return intrinsic value for invalid cases
            return np.where(is_call, intrinsic_call, intrinsic_put)
        
        # For very small sigma, return intrinsic directly to ensure consistency
        prices[mask_very_small_sigma & mask_valid] = np.where(
            is_call[mask_very_small_sigma & mask_valid],
            intrinsic_call[mask_very_small_sigma & mask_valid],
            intrinsic_put[mask_very_small_sigma & mask_valid]
        )
        
        # Calculate d1 and d2 only for valid cases with sigma >= threshold to avoid warnings
        # Skip BS calculation for very small sigma (already handled above)
        mask_bs_calc = mask_valid & ~mask_very_small_sigma
        d1 = np.full_like(S, np.nan, dtype=float)
        d2 = np.full_like(S, np.nan, dtype=float)
        
        if mask_bs_calc.any():
            d1_valid = BlackScholesEngine._d1(S[mask_bs_calc], K[mask_bs_calc], r, q, 
                                               sigma[mask_bs_calc], T[mask_bs_calc])
            d2_valid = BlackScholesEngine._d2(d1_valid, sigma[mask_bs_calc], T[mask_bs_calc])
            d1[mask_bs_calc] = d1_valid
            d2[mask_bs_calc] = d2_valid
        
        # Call prices (only for cases with sigma >= threshold)
        call_prices = np.full_like(S, np.nan, dtype=float)
        put_prices = np.full_like(S, np.nan, dtype=float)
        
        if mask_bs_calc.any():
            call_prices[mask_bs_calc] = (S[mask_bs_calc] * np.exp(-q * T[mask_bs_calc]) * norm.cdf(d1[mask_bs_calc]) - 
                                       K[mask_bs_calc] * np.exp(-r * T[mask_bs_calc]) * norm.cdf(d2[mask_bs_calc]))
            put_prices[mask_bs_calc] = (K[mask_bs_calc] * np.exp(-r * T[mask_bs_calc]) * norm.cdf(-d2[mask_bs_calc]) - 
                                      S[mask_bs_calc] * np.exp(-q * T[mask_bs_calc]) * norm.cdf(-d1[mask_bs_calc]))
        
        prices[mask_bs_calc] = np.where(is_call[mask_bs_calc], 
                                      call_prices[mask_bs_calc], 
                                      put_prices[mask_bs_calc])
        
        # Intrinsic value for invalid cases
        prices[~mask_valid] = np.where(is_call[~mask_valid], 
                                       intrinsic_call[~mask_valid], 
                                       intrinsic_put[~mask_valid])
        
        return prices


class ImpliedVolatilityCalculator:
    """
    Calculate implied volatility from market prices using Black-Scholes inversion.
    """
    
    def __init__(self, bs_engine: Optional[BlackScholesEngine] = None, ticker: Optional[str] = None):
        """
        Initialize the IV calculator.
        
        Args:
            bs_engine: Black-Scholes engine instance (creates new one if None)
            ticker: Ticker symbol for spot price fallback
        """
        self.bs_engine = bs_engine or BlackScholesEngine()
        self.ticker = ticker
    
    def implied_vol(self, is_call: bool, S: float, K: float, r: float, q: float,
                   T: float, price: float, lo: float = 1e-6, hi: float = 5.0) -> float:
        """
        Calculate implied volatility for a single option.
        
        Args:
            is_call: True for call, False for put
            S: Spot price
            K: Strike price
            r: Risk-free rate
            q: Dividend yield
            T: Time to expiration
            price: Market price
            lo: Lower bound for IV search
            hi: Upper bound for IV search
            
        Returns:
            Implied volatility (NaN if calculation fails)
        """
        # Validate inputs
        if S <= 0 or K <= 0 or T <= 0 or price < 0:
            logger.warning(f"IV calculation failed: invalid inputs (S={S}, K={K}, T={T}, price={price})")
            return np.nan
        
        # Check if price is below intrinsic value (no solution)
        intrinsic_call = max(0.0, S * exp(-q * T) - K * exp(-r * T))
        intrinsic_put = max(0.0, K * exp(-r * T) - S * exp(-q * T))
        intrinsic = intrinsic_call if is_call else intrinsic_put
        if price < intrinsic:
            logger.warning(f"IV calculation failed: price {price} below intrinsic {intrinsic}")
            return np.nan
        
        def f(sig: float) -> float:
            try:
                bs_price = self.bs_engine.price(
                    np.array([is_call]), np.array([S]), np.array([K]),
                    r, q, np.array([sig]), np.array([T])
                )[0]
                if not np.isfinite(bs_price):
                    return np.nan
                return bs_price - price
            except Exception:
                return np.nan
        
        try:
            flo = f(lo)
            if not np.isfinite(flo):
                logger.warning(f"IV calculation failed: f(lo) is not finite")
                return np.nan
            
            fhi = f(hi)
            if not np.isfinite(fhi):
                logger.warning(f"IV calculation failed: f(hi) is not finite")
                return np.nan
            
            if flo * fhi > 0:
                # Try expanding the upper bound
                for hi2 in (10.0, 20.0, 50.0):
                    fhi2 = f(hi2)
                    if np.isfinite(fhi2) and flo * fhi2 <= 0:
                        hi = hi2
                        fhi = fhi2
                        break
                else:
                    logger.warning(f"IV calculation failed: no root in [{lo}, {hi}] (flo={flo}, fhi={fhi})")
                    return np.nan
            
            return brentq(f, lo, hi, maxiter=200, xtol=1e-8)
        except (ValueError, RuntimeError) as e:
            logger.warning(f"IV calculation failed: {e}")
            return np.nan
        except Exception as e:
            logger.warning(f"IV calculation failed: {e}")
            return np.nan
    
    def infer_forward_from_parity(self, df_exp: pd.DataFrame, r: float) -> float:
        """
        Infer forward price from put-call parity.
        
        Args:
            df_exp: DataFrame with options for a single expiry
            r: Risk-free rate
            
        Returns:
            Forward price (NaN if calculation fails)
        """
        tmap = df_exp["type"].astype(str).str.lower().str[0]
        df_exp = df_exp.assign(_t=tmap)
        
        calls = df_exp[df_exp["_t"] == "c"].set_index("strike")
        puts = df_exp[df_exp["_t"] == "p"].set_index("strike")
        
        common = calls.index.intersection(puts.index)
        if len(common) == 0:
            return np.nan
        
        Cmid = calls.loc[common, "mid"]
        Pmid = puts.loc[common, "mid"]
        K = common.to_numpy(dtype=float)
        T = float(df_exp["T"].iloc[0])
        DF_r = np.exp(-r * T)
        
        Fvals = K + (Cmid.values - Pmid.values) / DF_r
        Fvals = Fvals[np.isfinite(Fvals)]
        
        return np.median(Fvals) if len(Fvals) > 0 else np.nan
    
    def recompute_clean_iv_surface(self, source_df: pd.DataFrame, 
                                   r: float = 0.045) -> pd.DataFrame:
        """
        Clean and compute implied volatility surface from raw options data.
        
        Args:
            source_df: Raw options DataFrame
            r: Risk-free rate
            
        Returns:
            Cleaned DataFrame with IV calculations
        """
        df = source_df.copy()
        
        # Validate required columns
        required = {"expiry", "strike"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"Missing columns: {missing}")
        
        # Clean expiry and calculate T
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
        df = df.dropna(subset=["expiry", "strike"])
        
        now_utc = datetime.now(timezone.utc)
        df["T"] = (df["expiry"].dt.normalize() - pd.Timestamp(now_utc.date())).dt.days.clip(lower=0) / 365.0
        
        # Standardize option type
        if "type" in df.columns:
            t = df["type"].astype(str).str.lower()
            df["type"] = np.where(t.str.startswith("c"), "call",
                         np.where(t.str.startswith("p"), "put", t))
        else:
            raise ValueError("Column 'type' not found")
        
        # Calculate mid prices
        for col in ("bid", "ask", "lastPrice", "last_price"):
            if col not in df.columns:
                df[col] = np.nan
        
        if "mid" not in df.columns:
            if df[["bid", "ask"]].notna().all(1).any():
                df["mid"] = (df["bid"] + df["ask"]) / 2
            else:
                df["mid"] = df.get("lastPrice", df.get("last_price", np.nan))
        
        df = df[df["mid"] > 0].copy()
        
        # Filter by spread
        if {"bid", "ask"}.issubset(df.columns):
            df["spread"] = df["ask"] - df["bid"]
            abs_cap = 0.25
            rel_cap = 0.80
            thr = np.where(df["mid"] < 1.0, abs_cap, rel_cap * df["mid"])
            df = df[(df["spread"] >= 0) & (df["spread"] <= thr)].copy()
        
        # Get spot price
        S = self._get_spot_price(df)
        df["S"] = S
        df["k"] = np.log(df["strike"] / S)
        
        # Calculate IV per expiry
        out = []
        for exp, g in df.groupby("expiry", sort=True):
            if g.empty:
                continue
            
            T = float(g["T"].iloc[0])
            if T <= 0:
                continue
            
            F = self.infer_forward_from_parity(g, r)
            if not np.isfinite(F):
                continue
            
            q = r - (1.0 / T) * np.log(max(F, 1e-12) / S)
            gg = g.copy()
            
            # Calculate IV for each option
            gg["iv_clean"] = gg.apply(
                lambda x: self.implied_vol(
                    x["type"] == "call", S, x["strike"], r, q, T, x["mid"]
                ),
                axis=1
            )
            
            # Filter reasonable IVs
            gg = gg[(gg["iv_clean"] > 0.0005) & (gg["iv_clean"] < 10.0)]
            out.append(gg)
        
        clean = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
        return clean
    
    def _get_spot_price(self, df: pd.DataFrame) -> float:
        """Extract or fetch spot price."""
        if "S" in df.columns and pd.notna(df["S"]).any():
            return float(pd.to_numeric(df["S"], errors="coerce").dropna().iloc[0])
        
        if "lastPrice" in df.columns and pd.notna(df["lastPrice"]).any():
            return float(pd.to_numeric(df["lastPrice"], errors="coerce").dropna().iloc[0])
        
        # Fallback to yfinance
        try:
            import yfinance as yf
            ticker_guess = self.ticker
            if not ticker_guess:
                raise RuntimeError("Cannot determine spot S. Provide ticker or S column.")
            hist = yf.Ticker(ticker_guess).history(period="1d")
            return float(hist["Close"].iloc[-1])
        except Exception as e:
            raise RuntimeError(f"Cannot determine spot S: {e}") from e

