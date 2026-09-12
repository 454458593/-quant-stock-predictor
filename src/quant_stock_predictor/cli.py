from __future__ import annotations

import argparse
from pathlib import Path

from .data import download_prices, normalize_ticker
from .features import FEATURE_COLUMNS, build_features, build_labeled_dataset
from .model import backtest, fit_full_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stock direction prediction research CLI")
    parser.add_argument("ticker", help="US ticker or six-digit A-share symbol")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument(
        "--adjust",
        choices=["hfq", "qfq", "none"],
        default="hfq",
        help="A-share adjustment: hfq (default), qfq, or none",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    adjust = "" if args.adjust == "none" else args.adjust
    prices = download_prices(args.ticker, args.start, args.end, adjust=adjust)
    features = build_features(prices).dropna(subset=FEATURE_COLUMNS)
    dataset = build_labeled_dataset(prices)
    result = backtest(dataset, args.train_ratio, args.threshold)

    full_model = fit_full_model(dataset)
    latest = features.iloc[[-1]]
    probability = float(full_model.predict_proba(latest[FEATURE_COLUMNS])[:, 1][0])

    args.output.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(args.output / "predictions.csv", encoding="utf-8-sig")
    result.predictions[["strategy_equity", "buy_hold_equity"]].to_csv(
        args.output / "equity_curve.csv", encoding="utf-8-sig"
    )

    symbol = normalize_ticker(args.ticker)
    print(f"Ticker: {symbol}")
    print(f"Data source: {prices.attrs.get('data_source', 'unknown')}")
    print(f"Latest feature date: {latest.index[-1].date()}")
    print(f"Next-session up probability: {probability:.2%}")
    signal = "LONG" if probability >= args.threshold else "CASH"
    print(f"Signal at threshold {args.threshold:.0%}: {signal}")
    print("Backtest metrics:")
    for name, value in result.metrics.items():
        if name == "test_observations":
            print(f"  {name}: {int(value)}")
        else:
            print(f"  {name}: {value:.2%}")


if __name__ == "__main__":
    main()
