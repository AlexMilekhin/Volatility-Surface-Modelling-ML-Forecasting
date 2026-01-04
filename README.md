# Volatility Surface Modeling & ML Forecasting

A professional Python package for volatility surface modeling, featuring SVI/SSVI fitting, risk-neutral density extraction, and deep learning forecasting.

## Package Structure

```
quant/
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data ingestion from Yahoo Finance/OpenBB
│   ├── features.py             # RND moment extraction (Skew, Kurtosis)
│   └── models/
│       ├── __init__.py
│       ├── bs_engine.py        # Black-Scholes pricing & IV calculation
│       ├── svi_fit.py          # SVI and SSVI surface fitting
│       └── forecaster.py       # LSTM + Self-Attention forecasting models
├── main.py                     # Command-line pipeline script
├── setup.py                    # Package installation script
└── requirements.txt           # Dependencies
```

## Installation

### Option 1: Install as package (recommended)

```bash
pip install -e .
```

### Option 2: Install dependencies only

```bash
pip install -r requirements.txt
```

## Usage

### Command-Line Pipeline

Run the complete pipeline from data collection to feature extraction:

```bash
python main.py --ticker JPM --data-dir data --metrics-dir metrics --rate 0.045
```

Arguments:
- `--ticker`: Stock ticker symbol (default: JPM)
- `--data-dir`: Directory for data files (default: data)
- `--metrics-dir`: Directory for metrics output (default: metrics)
- `--rate`: Risk-free rate (default: 0.045)

### Programmatic Usage

#### 1. Data Collection

```python
from src.data_loader import OptionsDataLoader

loader = OptionsDataLoader(ticker="JPM")
options_df = loader.fetch_options_chain()
hist_df = loader.fetch_historical_prices()
```

#### 2. IV Computation

```python
from src.models.bs_engine import ImpliedVolatilityCalculator

iv_calc = ImpliedVolatilityCalculator(ticker="JPM")
clean_df = iv_calc.recompute_clean_iv_surface(options_df, r=0.045)
```

#### 3. SVI Fitting

```python
from src.models.svi_fit import SVIFitter

svi_fitter = SVIFitter()
for expiry, df_exp in clean_df.groupby("expiry"):
    params, fitted_vols = svi_fitter.fit(df_exp)
    if params:
        iv_atm, skew_atm, curv_atm = SVIFitter.atm_metrics_from_params(params, T)
```

#### 4. SSVI Fitting

```python
from src.models.svi_fit import SSVIFitter

ssvi_fitter = SSVIFitter()
# Prepare thetas and forwards first
slice_params = ssvi_fitter.fit_global(clean_df, thetas, forwards, asof)
```

#### 5. RND Feature Extraction

```python
from src.features import RNDFeatureExtractor

rnd_extractor = RNDFeatureExtractor()
rnd_features = rnd_extractor.extract_features(ssvi_params_df, S0, r0)
```

#### 6. Deep Learning Forecasting

```python
from src.models.forecaster import LSTMSelfAttentionIVForecaster, IVForecasterTrainer
import torch

model = LSTMSelfAttentionIVForecaster(input_dim=20, hidden_dim=64, num_layers=2)
trainer = IVForecasterTrainer(model)
train_losses, val_losses = trainer.train(train_loader, val_loader, n_epochs=100)
```

## Module Details

### `src/data_loader.py`

- **OptionsDataLoader**: Fetches options chain data from OpenBB (with yfinance fallback)
- Handles data standardization and cleaning
- Saves data to parquet format

### `src/models/bs_engine.py`

- **BlackScholesEngine**: Vectorized Black-Scholes pricing
- **ImpliedVolatilityCalculator**: IV inversion using Brent's method
- Forward price inference from put-call parity
- Complete IV surface cleaning and computation

### `src/models/svi_fit.py`

- **SVIFitter**: Fits SVI model per expiry slice
- **SSVIFitter**: Global SSVI surface fitting with no-arbitrage constraints
- ATM metrics extraction (IV, skew, curvature)
- Monotone theta enforcement for calendar arbitrage reduction

### `src/features.py`

- **RNDFeatureExtractor**: Extracts risk-neutral density moments
- Computes skew, kurtosis, volatility from SSVI parameters
- Uses Breeden-Litzenberger formula for density extraction

### `src/models/forecaster.py`

- **LSTMSelfAttentionIVForecaster**: LSTM + self-attention model
- **IVForecasterTrainer**: Training loop with early stopping
- Weighted loss functions for quality-aware training
- Full evaluation pipeline

## Notebook Integration

The notebooks can now be "thin" - they should only handle visualization and orchestration:

```python
# In notebook: 1_data_collection.ipynb
from src.data_loader import OptionsDataLoader

loader = OptionsDataLoader("JPM")
df = loader.fetch_options_chain()
# ... visualization code ...
```

```python
# In notebook: 2_IV_compute.ipynb
from src.models.bs_engine import ImpliedVolatilityCalculator

iv_calc = ImpliedVolatilityCalculator(ticker="JPM")
clean_df = iv_calc.recompute_clean_iv_surface(df, r=0.045)
# ... plotting code ...
```

## Error Handling

All modules include comprehensive error handling:

- Data fetching failures are logged and raise exceptions
- IV calculations return NaN for invalid inputs
- Model fitting returns None on failure
- All functions use type hints for better IDE support

## Performance

- Vectorized NumPy operations throughout
- Efficient pandas groupby operations
- PyTorch GPU support for deep learning
- Optimized scipy optimization routines

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- pandas, numpy, scipy
- yfinance, openbb (optional)
- torch, scikit-learn
- pyarrow

## License

MIT License



