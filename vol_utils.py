import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm
from scipy.optimize import brentq

# --------------------------------------------
# 1. Black–Scholes core functions
# --------------------------------------------

def _d1(S, K, r, q, sigma, T):
    return (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))

def _d2(d1, sigma, T):
    return d1 - sigma * sqrt(T)

def bs_price(is_call, S, K, r, q, sigma, T):
    """Black–Scholes price for a European option."""
    if sigma <= 0 or T <= 0:
        return max(0.0, S * exp(-q * T) - K * exp(-r * T)) if is_call else max(
            0.0, K * exp(-r * T) - S * exp(-q * T)
        )
    d1 = _d1(S, K, r, q, sigma, T)
    d2 = _d2(d1, sigma, T)
    if is_call:
        return S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    else:
        return K * exp(-r * T) * norm.cdf(-d2) - S * exp(-q * T) * norm.cdf(-d1)

# --------------------------------------------
# 2. Implied volatility via Brent's method
# --------------------------------------------

def implied_vol(is_call, S, K, r, q, T, price, lo=1e-6, hi=5.0):
    """Numerically solve for Black–Scholes implied vol."""
    def f(sigma):
        return bs_price(is_call, S, K, r, q, sigma, T) - price

    try:
        if f(lo) * f(hi) < 0:
            return brentq(f, lo, hi, maxiter=100, xtol=1e-6)
    except Exception:
        pass
    return np.nan

# --------------------------------------------
# 3. Forward price inference from parity
# --------------------------------------------

def infer_forward_parity(df_expiry, r, atm_band=0.07):
    """
    Estimate forward price F = K + (C - P) / DF using call/put parity,
    restricted to near-the-money strikes.
    """
    df = df_expiry.copy()
    df = df[df["mid"].notna()]
    df["K"] = df["strike"]
    S_median = df["S"].median()
    near_atm = (df["K"] >= 0.7 * S_median) & (df["K"] <= 1.3 * S_median)
    df = df.loc[near_atm]
    calls = df[df["cp_flag"].str.upper() == "C"].set_index("K")
    puts  = df[df["cp_flag"].str.upper() == "P"].set_index("K")
    common_K = calls.index.intersection(puts.index)
    if len(common_K) == 0:
        return float(S_median)  # fallback

    C_mid = calls.loc[common_K, "mid"]
    P_mid = puts.loc[common_K, "mid"]
    T = df["T"].iloc[0]
    df_disc = exp(-r * T)
    F_est = common_K + (C_mid.values - P_mid.values) / df_disc
    return float(np.nanmedian(F_est[np.isfinite(F_est)])) if len(F_est) else float(S_median)