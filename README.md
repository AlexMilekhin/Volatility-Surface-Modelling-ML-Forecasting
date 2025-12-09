# Volatility Surface Modelling & ML Forecasting

An end-to-end quantitative pipeline for modeling equity volatility surfaces and forecasting dynamics using deep learning.

This project implements **Arbitrage-Free** volatility surface calibration using the **Surface SVI (SSVI)** parameterization, extracts high-order market beliefs (Skewness, Kurtosis) via **Risk-Neutral Density (RND)** integration, and utilizes these features to forecast future volatility with a PyTorch Neural Network.

## 🚀 Key Features

### 1\. Robust Data Pipeline

  * **Source:** Real-time option chains fetched via **OpenBB** and **Yahoo Finance** (underlying history).
  * **Cleaning:** Implements rigorous liquidity filtering (bid/ask \> 0, volume \> 0) and spread checks.
  * **Forward Price:** Infers the market-implied forward price via **Put-Call Parity** regression to ensure consistency across strikes.

### 2\. Arbitrage-Free Surface Calibration

  * **SVI & SSVI:** Fits raw implied volatilities to the **Stochastic Volatility Inspired (SVI)** model and its arbitrage-free extension, **Surface SVI (SSVI)**.
  * **Constraints:** Enforces calendar arbitrage constraints (total variance monotonicity) and static arbitrage constraints (Butterfly density).
  * **Optimization:** Uses non-linear least squares (L-BFGS-B/SLSQP) to minimize IV RMSE.

### 3\. Risk-Neutral Density (RND) Extraction

  * **Breeden-Litzenberger:** Numerically differentiates the calibrated option price surface to extract the market-implied probability distribution ($f(K) \approx \frac{\partial^2 C}{\partial K^2}$).
  * **Feature Engineering:** Calculates higher-order moments of the RND (Risk-Neutral Skewness & Kurtosis) to capture "crash risk" and "tail thickness" as ML features.

### 4\. Deep Learning Forecasting

  * **Architecture:** Feed-Forward Neural Network (MLP) built with **PyTorch**.
  * **Mechanics:** Uses **Batch Normalization** and **Adam** optimization to predict next-day ATM Volatility (`ATM_IV_{t+1}`).
  * **Input Features:** Term structure data, SSVI parameters ($\rho, \eta, \gamma$), and RND moments.

-----

## 📂 Repository Structure

| File / Notebook | Description |
|:---|:---|
| `1_data_collection.ipynb` | Fetches option chains for tickers (e.g., QQQ) and filters for liquidity. |
| `2_IV_compute.ipynb` | Cleans data, Inverts the Black-Scholes formula using **Brent's method** to compute Implied Volatility, plots IV surface. |
| `3_IV_analysis_SVI.ipynb` |Analyse IV surface (Skew, Term structure, Curvature), calibrate SVI using **quasi-explicit** method, compute smile features, plot SVI slices  |
| `4_ssvi_fit.ipynb` | Calibrates the SSVI surface to IV data (**Global fit**), enforcing no-arbitrage constraints (**Gatheral-Jacquier**). |
| `5_RND.ipynb` | Extracts the Risk-Neutral Density (**Breeden-Litzenberger**) from the surface and computes moments (Skew, Kurtosis) for feature engineering. |
| `6_ML_forecasting.ipynb` | Trains a PyTorch MLP to forecast volatility dynamics using the engineering features. |
| `vol_utils.py` | Helper library for Black-Scholes pricing, Greeks, and numerical optimization. |

## 🧠 Skills Demonstrated

Python · Quantitative Finance · Options Modelling · Implied Volatility Surfaces  
SVI / SSVI Calibration · Risk-Neutral Density Extraction · Numerical Optimisation  
Machine Learning (PyTorch) · Data Engineering · Arbitrage-Free Surface Construction
