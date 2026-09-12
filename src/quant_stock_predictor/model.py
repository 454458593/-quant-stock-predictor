from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS


@dataclass(frozen=True)
class BacktestResult:
    predictions: pd.DataFrame
    metrics: dict[str, float]
    model: Pipeline


def make_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
            ),
        ]
    )


def _performance_metrics(returns: pd.Series) -> dict[str, float]:
    clean = returns.dropna()
    equity = (1 + clean).cumprod()
    drawdown = equity / equity.cummax() - 1
    annual_return = float(equity.iloc[-1] ** (252 / len(clean)) - 1)
    volatility = float(clean.std(ddof=0) * np.sqrt(252))
    sharpe = annual_return / volatility if volatility > 0 else 0.0
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "annual_return": annual_return,
        "annual_volatility": volatility,
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "positive_day_rate": float((clean > 0).mean()),
    }


def backtest(
    dataset: pd.DataFrame,
    train_ratio: float = 0.7,
    threshold: float = 0.55,
) -> BacktestResult:
    """Fit on the first time block and evaluate only on the later block."""
    if not 0.5 <= train_ratio < 0.95:
        raise ValueError("train_ratio must be between 0.5 and 0.95")
    if not 0.5 <= threshold <= 1:
        raise ValueError("threshold must be between 0.5 and 1.0")
    if len(dataset) < 120:
        raise ValueError("At least 120 labeled trading days are required.")

    split = int(len(dataset) * train_ratio)
    train = dataset.iloc[:split]
    test = dataset.iloc[split:].copy()
    model = make_model()
    model.fit(train[FEATURE_COLUMNS], train["target"].astype(int))

    test["up_probability"] = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    test["position"] = (test["up_probability"] >= threshold).astype(int)
    test["strategy_return"] = test["position"] * test["future_return"]
    test["strategy_equity"] = (1 + test["strategy_return"]).cumprod()
    test["buy_hold_equity"] = (1 + test["future_return"]).cumprod()

    metrics = _performance_metrics(test["strategy_return"])
    metrics["buy_hold_total_return"] = float(test["buy_hold_equity"].iloc[-1] - 1)
    metrics["test_observations"] = float(len(test))
    return BacktestResult(predictions=test, metrics=metrics, model=model)


def fit_full_model(dataset: pd.DataFrame) -> Pipeline:
    model = make_model()
    model.fit(dataset[FEATURE_COLUMNS], dataset["target"].astype(int))
    return model
