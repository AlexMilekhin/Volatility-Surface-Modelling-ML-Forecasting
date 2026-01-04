import pandas as pd
import requests
import os
import numpy as np, pandas as pd
from datetime import datetime, timezone
from math import sqrt, exp, log
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
from scipy.optimize import least_squares, curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Implement SVI model fitting
def svi_variance(k, T, a, b, rho, m, sigma):
    """SVI total variance function."""
    return 0.5 * (a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))) * T

def svi_vol(k, T, a, b, rho, m, sigma):
    """SVI implied volatility function."""
    total_variance = svi_variance(k, T, a, b, rho, m, sigma)
    # Handle potential non-positive total variance (can happen with poor parameters)
    total_variance = np.maximum(1e-9, total_variance)
    return np.sqrt(total_variance / T)

def svi_objective(params, k, T, target_vol):
    """Objective function for SVI fitting (minimize squared errors)."""
    a, b, rho, m, sigma = params
    # Constraints check
    if sigma <= 0 or b < 0 or abs(rho) > 1:
         return np.inf * np.ones_like(k) # Penalize invalid parameters

    model_vol = svi_vol(k, T, a, b, rho, m, sigma)
    return (model_vol - target_vol)**2

def fit_svi(df_expiry):
    """Fits SVI model to data for a single expiry."""
    if df_expiry.empty: return None, None
    k = df_expiry["k"].values
    T = df_expiry["T"].iloc[0]
    target_vol = df_expiry["iv_clean"].values

    # Initial guess for SVI parameters
    a0 = np.min(target_vol)**2 * T * 2 # related to ATM variance
    b0 = 0.1 # slope
    rho0 = -0.5 # typical skew
    m0 = 0.0 # ATM
    sigma0 = 0.1 # width

    initial_guess = [a0, b0, rho0, m0, sigma0]

    # Bounds for parameters
    bounds = ([-np.inf, 0, -1, -np.inf, 1e-3], [np.inf, np.inf, 1, np.inf, np.inf])

    try:
        # Use least_squares with bounds
        result = least_squares(svi_objective, initial_guess, bounds=bounds, args=(k, T, target_vol), method='trf')
        if result.success:
            params = result.x
            # Calculate fitted vols
            fitted_vols = svi_vol(k, T, *params)
            return params, fitted_vols
        else:
            print(f"SVI fitting failed for expiry T={T:.3f}y: {result.message}")
            return None, None
    except Exception as e:
        print(f"Error during SVI fitting for expiry T={T:.3f}y: {e}")
        return None, None

# Fit models to each expiry and store results

if 'CLEAN' not in globals() or CLEAN.empty:
    print("CLEAN DataFrame not found or is empty. Please run the data cleaning cell first.")
else:
    # Convert 'expiry' column to datetime objects
    CLEAN['expiry'] = pd.to_datetime(CLEAN['expiry'])
    svi_fits = {}

    for expiry, df_expiry in CLEAN.groupby("expiry"):
        T = df_expiry["T"].iloc[0]
        # Access the date using the .date() method on the datetime object
        print(f"\nFitting models for expiry {expiry.date()} (T={T:.3f}y)")

        # Fit SVI
        svi_params, svi_fitted_vols = fit_svi(df_expiry)
        if svi_params is not None:
            svi_fits[expiry] = {
                "params": svi_params,
                "fitted_vols": svi_fitted_vols,
                "k": df_expiry["k"].values,
                "actual_vols": df_expiry["iv_clean"].values
            }
            print("SVI Params (a,b,rho,m,sigma):", np.round(svi_params, 4))
            # Simple visualization of SVI fit vs data
            plt.figure(figsize=(8, 4))
            plt.scatter(df_expiry["k"], df_expiry["iv_clean"], label="Actual IV", alpha=0.6)
            # Generate smooth curve for SVI
            k_smooth = np.linspace(df_expiry["k"].min(), df_expiry["k"].max(), 100)
            svi_smooth_vols = svi_vol(k_smooth, T, *svi_params)
            plt.plot(k_smooth, svi_smooth_vols, color='red', label='SVI Fit')
            plt.xlabel("log-moneyness k")
            plt.ylabel("Implied Volatility")
            plt.title(f"SVI Fit for Expiry {expiry.date()}")
            plt.legend()
            plt.grid(True)
            plt.show()

    print("\n✅ Model fitting process initiated. Check outputs for results per expiry.")
    print("SVI fits stored in 'svi_fits' dictionary.")

def svi_w(k, a,b,rho,m,sigma):
    return a + b*(rho*(k-m) + np.sqrt((k-m)**2 + sigma**2))

def svi_wprime(k, b,rho,m,sigma):
    return b*(rho + (k-m)/np.sqrt((k-m)**2 + sigma**2))

def svi_wpp(k, b,m,sigma):
    return b*(sigma**2)/(((k-m)**2 + sigma**2)**1.5)

def atm_metrics_from_row(params, T):
    """params = [a,b,rho,m,sig], returns (ATM_IV, ATM_skew, ATM_curv) at k=0"""
    a,b,rho,m,sigma = params
    k0 = 0.0
    w0  = svi_w(k0, a,b,rho,m,sigma)
    wp  = svi_wprime(k0, b,rho,m,sigma)
    wpp = svi_wpp(k0, b,m,sigma)
    iv_atm   = np.sqrt(max(w0, 0.0) / max(T, 1e-12))
    skew_atm = 0.5 * wp / np.sqrt(max(T*w0, 1e-12))
    curv_atm = 0.5/np.sqrt(max(T,1e-12)) * (
        - (wp**2) / (4*(w0**1.5 + 1e-18)) + (wpp / (2*np.sqrt(w0 + 1e-18)))
    )
    return float(iv_atm), float(skew_atm), float(curv_atm)

    