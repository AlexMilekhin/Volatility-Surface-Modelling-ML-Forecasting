"""
Data ingestion module for fetching and cleaning options data from Yahoo Finance/OpenBB.
"""

import logging
from typing import Optional, Tuple
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)


class OptionsDataLoader:
    """
    Handles fetching and cleaning options chain data from various sources.
    """
    
    def __init__(self, ticker: str, use_openbb: bool = True):
        """
        Initialize the data loader.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'JPM')
            use_openbb: Whether to use OpenBB for options data (default: True)
        """
        self.ticker = ticker
        self.use_openbb = use_openbb
        self.asof_date = datetime.now(timezone.utc).date()
        
    def fetch_options_chain(self) -> pd.DataFrame:
        """
        Fetch options chain data from OpenBB or yfinance.
        
        Returns:
            DataFrame with options chain data
            
        Raises:
            RuntimeError: If data fetching fails
        """
        try:
            if self.use_openbb:
                try:
                    from openbb import obb
                    chains = obb.derivatives.options.chains(symbol=self.ticker)
                    df = chains.to_dataframe()
                    logger.info(f"Fetched {len(df)} options from OpenBB for {self.ticker}")
                except ImportError:
                    logger.warning("OpenBB not available, falling back to yfinance")
                    df = self._fetch_from_yfinance()
                except Exception as e:
                    logger.warning(f"OpenBB fetch failed: {e}, falling back to yfinance")
                    df = self._fetch_from_yfinance()
            else:
                df = self._fetch_from_yfinance()
                
            # Standardize column names
            df = self._standardize_columns(df)
            
            # Add asof date
            df['asof'] = pd.to_datetime(self.asof_date)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch options data for {self.ticker}: {e}")
            raise RuntimeError(f"Data fetching failed: {e}") from e
    
    def _fetch_from_yfinance(self) -> pd.DataFrame:
        """Fallback to yfinance for options data."""
        try:
            ticker_obj = yf.Ticker(self.ticker)
            expirations = ticker_obj.options
            
            all_options = []
            for exp in expirations[:10]:  # Limit to first 10 expirations
                try:
                    opt_chain = ticker_obj.option_chain(exp)
                    calls = opt_chain.calls.copy()
                    calls['type'] = 'call'
                    calls['expiry'] = pd.to_datetime(exp)
                    
                    puts = opt_chain.puts.copy()
                    puts['type'] = 'put'
                    puts['expiry'] = pd.to_datetime(exp)
                    
                    all_options.append(calls)
                    all_options.append(puts)
                except Exception as e:
                    logger.warning(f"Failed to fetch expiration {exp}: {e}")
                    continue
            
            if not all_options:
                raise ValueError("No options data retrieved from yfinance")
                
            df = pd.concat(all_options, ignore_index=True)
            logger.info(f"Fetched {len(df)} options from yfinance for {self.ticker}")
            return df
            
        except Exception as e:
            logger.error(f"yfinance fetch failed: {e}")
            raise
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across different data sources."""
        # Rename common variations
        rename_map = {
            'expiration': 'expiry',
            'option_type': 'type',
            'lastPrice': 'last_price',
        }
        
        for old, new in rename_map.items():
            if old in df.columns:
                df.rename(columns={old: new}, inplace=True)
        
        # Ensure type column exists and is standardized
        if 'type' in df.columns:
            df['type'] = df['type'].astype(str).str.lower().str[0]
            df['type'] = df['type'].map({'c': 'call', 'p': 'put'})
        
        return df
    
    def fetch_historical_prices(self, period: str = "2y") -> pd.DataFrame:
        """
        Fetch historical underlying prices.
        
        Args:
            period: Time period for historical data (default: "2y")
            
        Returns:
            DataFrame with historical prices
        """
        try:
            ticker_obj = yf.Ticker(self.ticker)
            hist = ticker_obj.history(period=period)
            logger.info(f"Fetched {len(hist)} days of historical data for {self.ticker}")
            return hist
        except Exception as e:
            logger.error(f"Failed to fetch historical prices: {e}")
            raise RuntimeError(f"Historical price fetch failed: {e}") from e
    
    def save_data(self, options_df: pd.DataFrame, hist_df: Optional[pd.DataFrame] = None,
                  output_dir: str = "data") -> Tuple[str, Optional[str]]:
        """
        Save fetched data to parquet files.
        
        Args:
            options_df: Options chain DataFrame
            hist_df: Historical prices DataFrame (optional)
            output_dir: Output directory path
            
        Returns:
            Tuple of (options_path, hist_path)
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        options_path = f"{output_dir}/combined_options_data.parquet"
        options_df.to_parquet(options_path, index=False)
        logger.info(f"Saved options data to {options_path}")
        
        hist_path = None
        if hist_df is not None:
            hist_path = f"{output_dir}/hist_price.parquet"
            hist_df.to_parquet(hist_path)
            logger.info(f"Saved historical prices to {hist_path}")
        
        return options_path, hist_path

