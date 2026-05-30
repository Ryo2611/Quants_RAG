from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import EVENTS_CSV, PRICES_DIR, ensure_data_dirs
from src.event_loader import load_events


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [column[0] for column in df.columns]
    return df


def download_price_data(ticker: str, start_date: str, end_date: str, output_path: str | Path) -> pd.DataFrame:
    """Download daily adjusted price data using yfinance and save it as CSV."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if raw.empty:
        raise RuntimeError(f"No price data downloaded for {ticker} from {start_date} to {end_date}")
    raw = _flatten_columns(raw).reset_index()
    df = raw.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    expected = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    missing = [column for column in expected if column not in df.columns]
    if missing:
        raise ValueError(f"Downloaded price data for {ticker} is missing columns: {missing}")
    df = df[expected].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df.to_csv(output, index=False)
    return df


def load_price_data(path: str | Path) -> pd.DataFrame:
    """Load price CSV and return DataFrame with date as datetime."""
    price_path = Path(path)
    if not price_path.exists():
        raise FileNotFoundError(f"Price file not found: {price_path}")
    df = pd.read_csv(price_path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


def main() -> None:
    ensure_data_dirs()
    events = load_events(EVENTS_CSV)
    start = (events["earnings_date"].min() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    end = (events["earnings_date"].max() + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    for ticker in sorted(set(events["ticker"]) | {"SPY"}):
        output_path = PRICES_DIR / f"{ticker}.csv"
        df = download_price_data(ticker, start, end, output_path)
        print(f"Saved {len(df)} rows for {ticker} to {output_path}")


if __name__ == "__main__":
    main()
