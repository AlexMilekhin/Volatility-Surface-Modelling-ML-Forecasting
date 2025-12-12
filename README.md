# Volatility Surface Modelling & ML Forecasting

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Finance](https://img.shields.io/badge/Quant-Volatility%20Surface-green)

## 📖 Project Overview

This project implements a comprehensive pipeline for modeling and forecasting the Implied Volatility (IV) surface of equity options. It bridges the gap between traditional parametric quantitative finance models and modern deep learning techniques.

The core objective is to forecast future volatility surfaces by enhancing raw market data with structural insights derived from **Stochastic Volatility Inspired (SVI)** models and **Risk-Neutral Densities (RND)**.

### 🚀 Key Features
* **Automated Data Pipeline:** Fetches and processes option chain data using Yahoo Finance.
* **Surface Construction:** Computes Implied Volatility using Black-Scholes inversion.
* **Parametric Fitting:** Fits **SVI** and **Surface SVI (SSVI)** models to raw data to ensure arbitrage-free smoothing and interpolation.
* **Feature Engineering:** Extracts high-level economic indicators (Skewness, Kurtosis, Vol-of-Vol) from the Risk-Neutral Density derived from the fitted surfaces.
* **Deep Learning Forecast:** Implements an **LSTM with Self-Attention** mechanism to capture temporal dependencies and regime changes in the volatility surface.
* **Physics-Informed Weights:** Uses "Martingale Error" and fit costs to weight training samples, allowing the model to learn less from low-quality or arbitrage-violating market data.

## 🗂️ Repository Structure

The project is organized into modular notebooks representing the research workflow:

| File | Description |
| :--- | :--- |
| **`data_collection.ipynb`** | Scrapes and cleans historical option chain data from Yahoo Finance. |
| **`IV_compute.ipynb`** | Inverts the Black-Scholes formula to calculate raw Implied Volatility from option prices. |
| **`IV_analysis_SVI.ipynb`** | Performs initial analysis and fitting of the classic 5-parameter SVI model to individual slices. |
| **`ssvi_fit.ipynb`** | Fits the **Surface SVI (SSVI)** model to the entire surface simultaneously, enforcing calendar spread arbitrage constraints. |
| **`RND.ipynb`** | Derives the Risk-Neutral Probability Density Function (PDF) from the fitted volatility surface to extract higher-order moments (Skew, Kurtosis). |
| **`ML_forecasting.ipynb`** | The main forecasting playground. Prepares features, constructs time-series datasets, and trains baseline models. |
| **`LSTMSelfAttention.ipynb`** | **(Core Model)** Implements a custom PyTorch LSTM with a Self-Attention head to forecast IV changes, using a robust time-series validation split. |
| **`vol_utils.py`** | Helper library containing Black-Scholes pricing engines, Greeks calculations, and numerical optimization routines. |

## 🛠️ Methodology

1.  **Data Ingestion:** Raw option prices are filtered for liquidity and validity.
2.  **Calibration:** * The **SSVI** model is calibrated to market mid-prices to generate a smooth, arbitrage-free volatility surface $\sigma(k, \tau)$.
    * This step reduces noise and dimensionality compared to using raw option ticks.
3.  **Feature Extraction:** * Instead of feeding raw prices to the Neural Net, we feed it the *parameters* of the surface ($\rho, \eta, \theta$) and the *shape* of the market's expectation (RND Skew/Kurtosis).
4.  **Forecasting:**
    * An **LSTM** network processes the sequence of past surface states.
    * A **Self-Attention** mechanism highlights which historical time steps are most relevant for the current prediction.
    * **Weighted Loss:** The training loss is weighted by the inverse of the *Martingale Error*, ensuring the model focuses on economically valid data points.

## 💻 Tech Stack
* **Languages:** Python
* **Machine Learning:** PyTorch, Scikit-Learn
* **Quant Libraries:** SciPy (Optimization), NumPy, Pandas
* **Data Source:** `yfinance`
