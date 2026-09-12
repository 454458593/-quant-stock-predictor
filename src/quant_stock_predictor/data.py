from __future__ import annotations

import re

import pandas as pd


def normalize_ticker(ticker: str) -> str:
    """Normalize plain six-digit A-share symbols for Yahoo Finance."""
    value = ticker.strip().upper()
    if re.fullmatch(r"\d{6}", value):
        if value.startswith(("5", "6", "9")):
            return f"{value}.SS"
        return f"{value}.SZ"
    return value


def download_prices(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download adjusted daily OHLCV data and return a normalized frame."""
    import yfinance as yf

    symbol = normalize_ticker(ticker)
    frame = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        actions=False,
    )
    if frame.empty:
        raise ValueError(f"No market data returned for {symbol}.")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    result = frame[required].copy()
    result.index = pd.to_datetime(result.index)
    result = result.sort_index()
    result.attrs["ticker"] = symbol
    return result
