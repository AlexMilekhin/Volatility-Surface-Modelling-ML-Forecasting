from google.colab import drive
drive.mount('/content/drive')

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Display options (optional)
pd.set_option('display.max_columns', 50)

BASE_DIR = "/content/drive/MyDrive/Colab Notebooks/VolSurf_ML"
METRICS_DIR = os.path.join(BASE_DIR, "metrics")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# ---- Load SVI metrics ----
svi_files = sorted(glob.glob(os.path.join(METRICS_DIR, "svi_metrics_*.parquet")))
print("SVI files:")
for f in svi_files:
    print("  ", os.path.basename(f))

if not svi_files:
    raise RuntimeError("No svi_metrics_*.parquet files found in METRICS_DIR.")

svi_dfs = [pd.read_parquet(f) for f in svi_files]
svi_all = pd.concat(svi_dfs, ignore_index=True)

# Normalise types
svi_all['asof']   = pd.to_datetime(svi_all['asof'])
svi_all['expiry'] = pd.to_datetime(svi_all['expiry'])

# ---- Load SSVI metrics ----
ssvi_files = sorted(glob.glob(os.path.join(METRICS_DIR, "ssvi_metrics_*.parquet")))
print("\nSSVI files:")
for f in ssvi_files:
    print("  ", os.path.basename(f))

if not ssvi_files:
    raise RuntimeError("No ssvi_metrics_*.parquet files found in METRICS_DIR.")

ssvi_dfs = [pd.read_parquet(f) for f in ssvi_files]
ssvi_all = pd.concat(ssvi_dfs, ignore_index=True)

ssvi_all['asof']   = pd.to_datetime(ssvi_all['asof'])
ssvi_all['expiry'] = pd.to_datetime(ssvi_all['expiry'])

# ---- Merge SVI + SSVI on (asof, expiry, T) ----
# Keep only SSVI columns we care about
ssvi_cols = ['asof','expiry','T','theta','ssvi_rho','ssvi_eta','ssvi_p','fit_cost']
if 'calendar_violations' in ssvi_all.columns:
    ssvi_cols.append('calendar_violations')

metrics_all = svi_all.merge(
    ssvi_all[ssvi_cols],
    on=['asof','expiry'],
    how='left',
    suffixes=('', '_ssvi')
)

metrics_all = metrics_all.sort_values(['asof', 'T']).reset_index(drop=True)

# ---- Helper Function to Flatten Array-Wrapped Cells ----
def clean_flatten_cols(df):
    """
    Detects columns containing single-element arrays (e.g. [0.035])
    and flattens them to scalars (0.035).
    """
    for col in df.columns:
        # Skip columns that are already standard floats/ints/dates
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        # Check a non-null sample to see if it's an array/list
        valid_vals = df[col].dropna()
        if valid_vals.empty:
            continue

        sample = valid_vals.iloc[0]
        if isinstance(sample, (np.ndarray, list)):
            print(f"  -> Flattening array-wrapped column: {col}")
            # Extract item() if strictly numpy, or [0] if list-like
            df[col] = df[col].apply(lambda x: x.item() if hasattr(x, 'item') else (x[0] if hasattr(x, '__getitem__') and len(x) > 0 else np.nan))
            # Force to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

# ---- Load RND Features ----
rnd_files = sorted(glob.glob(os.path.join(METRICS_DIR, "rnd_features_*.parquet")))

if rnd_files:
    print(f"Found {len(rnd_files)} RND feature files.")
    rnd_dfs = [pd.read_parquet(f) for f in rnd_files]
    rnd_all = pd.concat(rnd_dfs, ignore_index=True)

    # --- STEP 1: CLEANING (The Fix) ---
    print("Cleaning RND data...")
    rnd_all = clean_flatten_cols(rnd_all)

    # --- STEP 2: DATE NORMALIZATION ---
    if 'asof' not in rnd_all.columns and 'date' in rnd_all.columns:
        rnd_all.rename(columns={'date': 'asof'}, inplace=True)

    if 'asof' in rnd_all.columns:
        rnd_all['asof'] = pd.to_datetime(rnd_all['asof'])
        if rnd_all['asof'].dt.tz is not None:
            rnd_all['asof'] = rnd_all['asof'].dt.tz_localize(None)
        rnd_all['asof'] = rnd_all['asof'].dt.normalize()

    if 'expiry' in rnd_all.columns:
        rnd_all['expiry'] = pd.to_datetime(rnd_all['expiry']).dt.normalize()

    # --- STEP 3: MERGE ---
    # Ensure metrics_all dates are also normalized for matching
    metrics_all['asof'] = pd.to_datetime(metrics_all['asof'])
    if metrics_all['asof'].dt.tz is not None:
        metrics_all['asof'] = metrics_all['asof'].dt.tz_localize(None)
    metrics_all['asof'] = metrics_all['asof'].dt.normalize()
    metrics_all['expiry'] = pd.to_datetime(metrics_all['expiry']).dt.normalize()

    # Drop 'T' from rnd to avoid duplicates/floating point mismatches
    cols_to_use = [c for c in rnd_all.columns if c != 'T']

    # Merge on Date Keys
    metrics_all = metrics_all.merge(
        rnd_all[cols_to_use],
        on=['asof', 'expiry'],
        how='left',
        suffixes=('', '_rnd')
    )

    # --- STEP 4: FILL MISSING RND ---
    # Fallback for days where RND might have failed but SVI existed
    if 'rnd_skew' in metrics_all.columns:
        metrics_all['rnd_skew'] = metrics_all['rnd_skew'].fillna(metrics_all['ATM_skew'])
    if 'rnd_kurtosis' in metrics_all.columns:
        metrics_all['rnd_kurtosis'] = metrics_all['rnd_kurtosis'].fillna(3.0)
    if 'martingale_error' in metrics_all.columns:
        # Fill missing martingale errors with a "high confidence" or "average" value?
        # Usually better to fill with 0.0 (assume okay) or the mean.
        metrics_all['martingale_error'] = metrics_all['martingale_error'].fillna(0.0)

    print(f"RND merged and cleaned. New shape: {metrics_all.shape}")
else:
    print("No RND files found.")


print("\nCombined metrics shape:", metrics_all.shape)


display(metrics_all)

price_path = os.path.join(BASE_DIR, "hist_price.parquet")
prices = pd.read_parquet(price_path)

# Ensure index is sorted
prices = prices.sort_index()

# If index is timezone-aware (like 2023-11-15 00:00:00-05:00), drop tz
if hasattr(prices.index, "tz") and prices.index.tz is not None:
    # Option 1: convert to UTC then drop tz
    prices.index = prices.index.tz_convert("UTC").tz_localize(None)
    # Option 2 (simpler, also fine here): prices.index = prices.index.tz_localize(None)

# Basic returns and realised vol
prices['return_1d'] = prices['Close'].pct_change()
prices['return_5d'] = prices['Close'].pct_change(5)
prices['rv_20d']    = prices['return_1d'].rolling(20).std() * np.sqrt(252)

# Turn index into a column called 'asof' (date-like, no tz)
prices = prices.reset_index().rename(columns={prices.index.name or 'Date': 'asof'})
prices['asof'] = pd.to_datetime(prices['asof']).dt.normalize()

prices.tail()

df = metrics_all.merge(
    prices[['asof', 'Close', 'return_1d', 'return_5d', 'rv_20d']],
    on='asof',
    how='left'
)

df = df.rename(columns={'Close': 'spot'})
df = df.sort_values(['T', 'asof']).reset_index(drop=True)

print("Merged shape:", df.shape)
df.tail()

df = df.sort_values(['T', 'asof']).reset_index(drop=True)

# Shift ATM_IV one day ahead within each tenor
df['ATM_IV_tplus1'] = df.groupby('T')['ATM_IV'].shift(-1)

missing_mask = df['ATM_IV_tplus1'].isna()
df.loc[missing_mask, 'ATM_IV_tplus1'] = df.loc[missing_mask, 'ATM_IV']

print("After creating target, shape:", df.shape)
df[['asof', 'T', 'ATM_IV', 'ATM_IV_tplus1']].tail(10)

feature_cols = [
    'T',
    # ATM + SVI
    'ATM_IV', 'ATM_skew', 'ATM_curvature',
    'a', 'b', 'rho', 'm', 'sigma',
    # SSVI
    'theta', 'ssvi_rho', 'ssvi_eta', 'ssvi_p',

    #RND

    'rnd_vol', 'rnd_skew', "rnd_kurtosis", 'martingale_error',
    # optional: calendar / fit diagnostics
    # 'fit_cost',
    # 'calendar_violations',
    # market features (when merge_asof is fixed)
     'return_1d', 'rv_20d'
]

target_col = 'ATM_IV_tplus1'

data = df.dropna(subset=feature_cols + [target_col]).copy()
data = data.sort_values('asof').reset_index(drop=True)
df = df.sort_values(['asof', 'T']).reset_index(drop=True)
print("Final usable rows:", len(data))
data[ ['asof', 'T', 'ATM_IV', 'ATM_IV_tplus1'] + feature_cols ].head()

n = len(data)
if n < 30:
    print("⚠️ Warning: very few samples for ML. The notebook will run, but forecasting quality will be limited until you collect more days.")

train_end = int(n * 0.7)
val_end   = int(n * 0.85)

train = data.iloc[:train_end]
val   = data.iloc[train_end:val_end]
test  = data.iloc[val_end:]

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

X_train = train[feature_cols].values.astype(np.float32)
y_train = train[target_col].values.astype(np.float32)

X_val   = val[feature_cols].values.astype(np.float32)
y_val   = val[target_col].values.astype(np.float32)

X_test  = test[feature_cols].values.astype(np.float32)
y_test  = test[target_col].values.astype(np.float32)

dates_test = test['asof'].values

def get_weights(df, col='martingale_error', scale=100.0):
  err = df[col].fillna(df[col].mean()).values.astype(np.float32)
  return 1.0 / (1.0 + scale *err)

w_train = get_weights(train)
w_val = get_weights(val)
w_test = get_weights(test)

if len(test) > 0:
    y_pred_persist = test['ATM_IV'].values.astype(np.float32)

    mae_persist = mean_absolute_error(y_test, y_pred_persist)
    rmse_persist = mean_squared_error(y_test, y_pred_persist)
    print(f"Baseline (persistence) – MAE: {mae_persist:.6f}, RMSE: {rmse_persist:.6f}")
else:
    print("No test set; cannot compute baseline yet.")

def make_loader(X, y, w, batch_size=64, shuffle=False):
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y).unsqueeze(1)  # (N,) -> (N,1)
    w_tensor = torch.from_numpy(w).unsqueeze(1) #match y shape
    ds = TensorDataset(X_tensor, y_tensor, w_tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_train, y_train, w_train, batch_size=64, shuffle=True)
val_loader   = make_loader(X_val,   y_val, w_val, batch_size=256, shuffle=False)
test_loader  = make_loader(X_test,  y_test,  w_test, batch_size=256, shuffle=False)

input_dim = X_train.shape[1]

class VolForecastNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, input_dim)

        # Layer 1: Linear -> BatchNorm -> ReLU
        x = self.fc1(x)
        x = self.bn1(x)   # <-- BatchNorm applied here
        x = self.act(x)

        # Layer 2: Linear -> BatchNorm -> ReLU
        x = self.fc2(x)
        x = self.bn2(x)   # <-- BatchNorm applied here
        x = self.act(x)

        # Output layer: Linear (no BatchNorm, no activation)
        x = self.fc3(x)
        return x

model = VolForecastNet(input_dim).to(device)
criterion = nn.HuberLoss(delta=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model

best_val_loss = float('inf')
best_state = None
patience = 10
patience_counter = 0

train_history = []
val_history = []

for epoch in range(100):  # max epochs
    # ---- Train ----
    model.train()
    batch_losses = []
    for xb, yb, wb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        wb = wb.to(device)

        optimizer.zero_grad()
        preds = model(xb)
        raw_loss = (preds-yb)**2
        weighted_loss = raw_loss * wb
        loss = weighted_loss.mean()
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    train_loss = np.mean(batch_losses)

    # ---- Validate ----
    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb, wb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            preds = model(xb)
            val_loss_batch = (wb*(preds-yb)**2).mean()
            val_losses.append(val_loss_batch.item())

    val_loss = np.mean(val_losses) if val_losses else np.nan

    train_history.append(train_loss)
    val_history.append(val_loss)

    print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}")

    # Early stopping
    if val_losses and val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        best_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping.")
            break

# Load best model (if we have one)
if best_state is not None:
    model.load_state_dict(best_state)

model.eval()
y_pred_list = []
y_true_list = []

with torch.no_grad():
    for xb, yb, wb in test_loader:
        xb = xb.to(device)

        # Get raw model predictions
        preds = model(xb).cpu().numpy().ravel()
        y_pred_list.append(preds)
        y_true_list.append(yb.numpy().ravel())

if y_pred_list:
    y_pred_nn = np.concatenate(y_pred_list)
    y_true_eval = np.concatenate(y_true_list) # Use targets from loader to ensure alignment

    # Calculate metrics
    mae_nn = mean_absolute_error(y_true_eval, y_pred_nn)
    rmse_nn = mean_squared_error(y_true_eval, y_pred_nn)

    print(f"PyTorch MLP – MAE: {mae_nn:.6f}, RMSE: {rmse_nn:.6f}")

    # Optional: Compare with persistence if available
    if 'mae_persist' in locals():
        print(f"Baseline     – MAE: {mae_persist:.6f}, RMSE: {rmse_persist:.6f}")
else:
    print("No predictions from test_loader (probably empty test set).")

if len(test) > 0 and y_pred_list:
    plt.figure(figsize=(12,5))
    plt.plot(dates_test, y_test, label='Actual ATM_IV_{t+1}', marker='o')
    plt.plot(dates_test, y_pred_persist, label='Persistence', linestyle='--')
    plt.plot(dates_test, y_pred_nn, label='PyTorch MLP', linestyle='-.')
    plt.xlabel("Date")
    plt.ylabel("ATM_IV_{t+1}")
    plt.title("Next-day ATM IV Forecast – Actual vs Models")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Not enough test data to plot.")