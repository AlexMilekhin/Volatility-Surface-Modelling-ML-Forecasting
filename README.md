# Volatility Surface Modelling & ML Forecasting

This project builds an end-to-end **options analytics system** using Python and real market data from Yahoo Finance.  
It visualises implied volatility surfaces, fits SVI parameters, and applies **machine learning** to forecast volatility dynamics.

## 📁 Structure
| Notebook | Description |
|-----------|--------------|
| `1_data_collection.ipynb` | Collect option-chain data |
| `2_IV_compute.ipynb` | Clean data and compute IV's |
| `3_IV_Analysis_SVI.ipynb` | Compute IVs, features, fit SVI, plot surfaces |
| `4_ssvi_fit.ipynb` | Compute SSVI, monotone theta, diagnostics, time interpolation, plot surface |
| `5_ML_forecasting.ipynb` | Extract Vol, SVI, SSVI features, implements PyTorch MLP with BatchNorm + Adam|

## 📊 Highlights
- Dynamic SVI calibration
- Volatility term-structure analysis
- Real-data and synthetic surface generation
- PyTorch MLP with BatchNorm and Adam optimisation
- Polished visual outputs (Matplotlib/Plotly)

## 🧠 Skills Demonstrated
Python · Quantitative Finance · Machine Learning · Data Engineering · Options Modelling 
