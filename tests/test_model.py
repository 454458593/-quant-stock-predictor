import numpy as np
from test_features import sample_prices

from quant_stock_predictor.features import build_labeled_dataset
from quant_stock_predictor.model import backtest


def test_backtest_returns_finite_metrics() -> None:
    result = backtest(build_labeled_dataset(sample_prices(260)), threshold=0.5)
    assert len(result.predictions) > 0
    assert {"up_probability", "position", "strategy_return"}.issubset(
        result.predictions.columns
    )
    assert all(np.isfinite(value) for value in result.metrics.values())
