from __future__ import annotations

import re

import pandas as pd


def a_share_code(ticker: str) -> str | None:
    """Return a six-digit A-share symbol, accepting optional Yahoo suffixes."""
    value = ticker.strip().upper()
    match = re.fullmatch(r"(\d{6})(?:\.(?:SS|SZ|BJ))?", value)
    return match.group(1) if match else None


def normalize_ticker(ticker: str) -> str:
    """Return a readable market-qualified symbol."""
    value = ticker.strip().upper()
    code = a_share_code(value)
    if code:
        if code.startswith(("4", "8")):
            return f"{code}.BJ"
        if code.startswith(("5", "6", "9")):
            return f"{code}.SS"
        return f"{code}.SZ"
    return value


def _normalize_price_frame(frame: pd.DataFrame, symbol: str, source: str) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"No market data returned for {symbol}.")
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    result = frame[required].copy()
    result.index = pd.to_datetime(result.index)
    result = result.apply(pd.to_numeric, errors="coerce").dropna(subset=required)
    result = result.sort_index()
    result.attrs["ticker"] = symbol
    result.attrs["data_source"] = source
    return result


def _download_a_share(
    code: str,
    start: str,
    end: str | None,
    adjust: str,
) -> pd.DataFrame:
    import akshare as ak

    start_date = pd.Timestamp(start).strftime("%Y%m%d")
    end_date = pd.Timestamp(end or pd.Timestamp.today()).strftime("%Y%m%d")
    frame = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )
    frame = frame.rename(
        columns={
            "日期": "Date",
            "开盘": "Open",
            "最高": "High",
            "最低": "Low",
            "收盘": "Close",
            "成交量": "Volume",
        }
    )
    if "Date" not in frame.columns:
        raise ValueError("AKShare response is missing the 日期 column.")
    frame = frame.set_index("Date")
    return _normalize_price_frame(frame, normalize_ticker(code), "akshare")


def _download_us_share(ticker: str, start: str, end: str | None) -> pd.DataFrame:
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
    return _normalize_price_frame(frame, symbol, "yfinance")


def download_prices(
    ticker: str,
    start: str,
    end: str | None = None,
    adjust: str = "hfq",
) -> pd.DataFrame:
    """Use AKShare for A-shares and yfinance for other tickers."""
    if adjust not in {"", "qfq", "hfq"}:
        raise ValueError("adjust must be one of: '', 'qfq', 'hfq'")
    code = a_share_code(ticker)
    if code:
        return _download_a_share(code, start, end, adjust)
    return _download_us_share(ticker, start, end)
