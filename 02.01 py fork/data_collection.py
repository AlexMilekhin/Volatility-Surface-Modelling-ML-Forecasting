
!pip install openbb

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import requests
import os
import numpy as np, pandas as pd
from datetime import datetime, timezone
from math import sqrt, exp, log
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import yfinance as yf
from openbb import obb

TICKER='QQQ'

get_chains = obb.derivatives.options.chains(symbol=TICKER)

ALL = get_chains.to_dataframe()
ALL.rename(columns={'expiration': 'expiry'}, inplace=True)
ALL.rename(columns={'option_type':'type'}, inplace = True)

ASOF = datetime.now(timezone.utc).date()
ALL['asof']=pd.to_datetime(ASOF)

display(ALL.head(10))

"""As most of the data is untradable, moving forward we will only use rows with bid, ask and volume > 0"""

ALL = ALL[(ALL["bid"] > 0) & (ALL["ask"] > 0) & (ALL["volume"] > 0)]

try:
    ALL.to_parquet('/content/drive/MyDrive/Colab Notebooks/VolSurf_ML/combined_options_data.parquet', index=False)
    print("Successfully saved combined_options_data.parquet to Google Drive.")
except Exception as e:
    print(f"Error saving file to Google Drive: {e}")
    print("Make sure you have mounted your Google Drive correctly.")

# Underlying historical prices (for realised vol, returns, etc.)
hist_price = yf.Ticker(TICKER).history(period="2y")  # choose horizon
hist_price_path = '/content/drive/MyDrive/Colab Notebooks/VolSurf_ML/hist_price.parquet'
hist_price.to_parquet(hist_price_path)
print(f"Saved underlying history to {hist_price_path}")

print("Total rows:", len(ALL))

print("\nBid/ask quality:")
print("  both bid>0, ask>0:", ((ALL["bid"] > 0) & (ALL["ask"] > 0)).sum())
print("  only bid>0:", ((ALL["bid"] > 0) & (ALL["ask"] == 0)).sum())
print("  only ask>0:", ((ALL["bid"] == 0) & (ALL["ask"] > 0)).sum())
print("  both 0:", ((ALL["bid"] == 0) & (ALL["ask"] == 0)).sum())

print("\nVolume / OI:")
print("  volume>0:", (ALL["volume"] > 0).sum())
print("  open_interest>0:", (ALL["open_interest"] > 0).sum())

print("\nImplied vol nonzero:", (ALL["implied_volatility"] > 0).sum())

tradable = ALL[(ALL["bid"] > 0) & (ALL["ask"] > 0)]
print("Tradable rows:", len(tradable))
print(tradable.groupby("dte").size().head(20))   # or groupby("expiry") if you’ve converted

