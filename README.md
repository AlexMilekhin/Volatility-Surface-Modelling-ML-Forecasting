# Volatility Surface Modelling & ML Forecasting

This project builds an end-to-end **options analytics system** using Python and real market data from Yahoo Finance.  
It visualises implied volatility surfaces, fits SVI parameters, and applies **machine learning** to forecast volatility dynamics.

## 📁 Structure
| Notebook | Description |
|-----------|--------------|
| `1_data_collection.ipynb` | Collect option-chain data |
| `2_IV_compute.ipynb` | Clean data and compute IV's |
| `3_IV_Analysis_SVI.ipynb` | Compute IVs, features, fit SVI, plot surfaces |
| `4_ml_forecasting.ipynb` | Predict next-day ATM IV change using ML |
| `utils/helpers.py` | Shared functions for BS pricing and plotting |

## 📊 Highlights
- Dynamic SVI calibration
- Volatility term-structure analysis
- Real-data and synthetic surface generation
- Polished visual outputs (Matplotlib/Plotly)

## 🧠 Skills Demonstrated
Python · Quantitative Finance · Machine Learning · Data Engineering · Options Modelling
