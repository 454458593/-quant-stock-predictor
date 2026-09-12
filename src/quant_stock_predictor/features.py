from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "ma_5_ratio",
    "ma_20_ratio",
    "volatility_20d",
    "rsi_14",
    "volume_ratio_20d",
    "range_ratio",
]


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build features using information available no later than each row's close."""
    close = prices["Close"].astype(float)
    volume = prices["Volume"].astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    relative_strength = gain / loss.replace(0, np.nan)

    features = pd.DataFrame(index=prices.index)
    features["return_1d"] = close.pct_change()
    features["return_5d"] = close.pct_change(5)
    features["ma_5_ratio"] = close / close.rolling(5).mean() - 1
    features["ma_20_ratio"] = close / close.rolling(20).mean() - 1
    features["volatility_20d"] = features["return_1d"].rolling(20).std()
    features["rsi_14"] = 100 - 100 / (1 + relative_strength)
    features["volume_ratio_20d"] = volume / volume.rolling(20).mean()
    features["range_ratio"] = (prices["High"] - prices["Low"]) / close
    return features.replace([np.inf, -np.inf], np.nan)


def build_labeled_dataset(prices: pd.DataFrame) -> pd.DataFrame:
    """Attach next-session return and direction labels without filling the final row."""
    features = build_features(prices)
    future_return = prices["Close"].shift(-1) / prices["Close"] - 1
    dataset = features.copy()
    dataset["future_return"] = future_return
    dataset["target"] = np.where(
        future_return.notna(), (future_return > 0).astype(float), np.nan
    )
    return dataset.dropna(subset=FEATURE_COLUMNS + ["future_return", "target"])
