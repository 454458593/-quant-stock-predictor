import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from quant_stock_predictor.data import download_prices, normalize_ticker
from quant_stock_predictor.features import FEATURE_COLUMNS, build_features, build_labeled_dataset


def sample_prices(rows: int = 180) -> pd.DataFrame:
    index = pd.bdate_range("2025-01-01", periods=rows)
    close = pd.Series(100 + np.linspace(0, 20, rows) + np.sin(np.arange(rows)), index=index)
    return pd.DataFrame(
        {
            "Open": close * 0.998,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000 + np.arange(rows) * 100,
        },
        index=index,
    )


def test_normalize_ticker() -> None:
    assert normalize_ticker("600519") == "600519.SS"
    assert normalize_ticker("000001") == "000001.SZ"
    assert normalize_ticker("830799") == "830799.BJ"
    assert normalize_ticker("aapl") == "AAPL"


def test_a_share_download_uses_akshare(monkeypatch) -> None:
    calls = {}

    def stock_zh_a_hist(**kwargs):
        calls.update(kwargs)
        return pd.DataFrame(
            {
                "日期": ["2025-01-02", "2025-01-03"],
                "开盘": [10.0, 10.2],
                "最高": [10.5, 10.4],
                "最低": [9.9, 10.0],
                "收盘": [10.2, 10.3],
                "成交量": [1000, 1200],
            }
        )

    monkeypatch.setitem(
        sys.modules, "akshare", SimpleNamespace(stock_zh_a_hist=stock_zh_a_hist)
    )
    prices = download_prices("600519", "2025-01-01", "2025-01-31", adjust="hfq")
    assert calls == {
        "symbol": "600519",
        "period": "daily",
        "start_date": "20250101",
        "end_date": "20250131",
        "adjust": "hfq",
    }
    assert list(prices.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert prices.attrs["data_source"] == "akshare"


def test_features_use_expected_columns() -> None:
    features = build_features(sample_prices())
    assert list(features.columns) == FEATURE_COLUMNS
    assert len(features.dropna()) > 100


def test_last_unknown_target_is_removed() -> None:
    prices = sample_prices()
    dataset = build_labeled_dataset(prices)
    assert dataset.index.max() < prices.index.max()
    assert dataset["target"].isin([0.0, 1.0]).all()
