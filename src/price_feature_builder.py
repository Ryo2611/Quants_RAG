from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import EVENTS_CSV, PRICE_FEATURES_CSV, PRICES_DIR, ensure_data_dirs
from src.event_loader import load_events
from src.price_loader import load_price_data
from src.target_builder import get_next_trading_day


def _window_before_event(price_df: pd.DataFrame, event_date: str | pd.Timestamp, lookback: int) -> pd.DataFrame:
    dates = price_df["date"].sort_values().reset_index(drop=True)
    event_day = get_next_trading_day(price_df, event_date)
    event_index = int(dates[dates == event_day].index[0])
    start_index = max(0, event_index - lookback)
    return price_df[price_df["date"].isin(dates.iloc[start_index:event_index])].copy()


def _pre_return(price_df: pd.DataFrame, event_date: str | pd.Timestamp, lookback: int) -> float:
    window = _window_before_event(price_df, event_date, lookback)
    if len(window) < 2:
        return np.nan
    return float(window["close"].iloc[-1] / window["close"].iloc[0] - 1.0)


def compute_pre_event_features(price_df: pd.DataFrame, market_df: pd.DataFrame, event_date: str | pd.Timestamp) -> dict:
    """Compute pre-event stock and market features using only information before earnings_date."""
    stock_window_20 = _window_before_event(price_df, event_date, 20)
    volume_5 = stock_window_20["volume"].tail(5).mean() if not stock_window_20.empty else np.nan
    volume_20 = stock_window_20["volume"].mean() if not stock_window_20.empty else np.nan
    daily_returns = stock_window_20["close"].pct_change().dropna()
    return {
        "pre_return_5d": _pre_return(price_df, event_date, 5),
        "pre_return_20d": _pre_return(price_df, event_date, 20),
        "pre_volatility_20d": float(daily_returns.std()) if not daily_returns.empty else np.nan,
        "pre_volume_change": float(volume_5 / volume_20 - 1.0) if volume_20 and not np.isnan(volume_20) else np.nan,
        "market_pre_return_5d": _pre_return(market_df, event_date, 5),
        "market_pre_return_20d": _pre_return(market_df, event_date, 20),
    }


def build_price_features(events_df: pd.DataFrame, stock_prices: pd.DataFrame, market_prices: pd.DataFrame) -> pd.DataFrame:
    """Create price feature table for each earnings event."""
    rows = []
    for _, event in events_df.iterrows():
        row = {"event_id": event["event_id"], "ticker": event["ticker"], "earnings_date": event["earnings_date"]}
        row.update(compute_pre_event_features(stock_prices, market_prices, event["earnings_date"]))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ensure_data_dirs()
    events = load_events(EVENTS_CSV)
    ticker = str(events["ticker"].iloc[0])
    features = build_price_features(events, load_price_data(PRICES_DIR / f"{ticker}.csv"), load_price_data(PRICES_DIR / "SPY.csv"))
    Path(PRICE_FEATURES_CSV).parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(PRICE_FEATURES_CSV, index=False)
    print(f"Saved price features to {PRICE_FEATURES_CSV}")


if __name__ == "__main__":
    main()
