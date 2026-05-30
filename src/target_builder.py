from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import EVENTS_CSV, FEATURES_DIR, PRICES_DIR, TARGETS_CSV, ensure_data_dirs
from src.event_loader import load_events
from src.price_loader import load_price_data


def get_next_trading_day(price_df: pd.DataFrame, date: str | pd.Timestamp) -> pd.Timestamp:
    """Return the first trading day after or equal to the given date."""
    event_date = pd.to_datetime(date).normalize()
    dates = price_df["date"].sort_values().reset_index(drop=True)
    matches = dates[dates >= event_date]
    if matches.empty:
        raise ValueError(f"No trading day found on or after {event_date.date()}")
    return pd.Timestamp(matches.iloc[0])


def compute_forward_return(price_df: pd.DataFrame, event_date: str | pd.Timestamp, horizon: int = 5) -> float:
    """Compute close-to-close forward return from t+1 to t+horizon."""
    dates = price_df["date"].sort_values().reset_index(drop=True)
    start_day = get_next_trading_day(price_df, event_date)
    event_index = int(dates[dates == start_day].index[0])
    start_index = event_index + 1
    end_index = event_index + horizon
    if end_index >= len(dates) or start_index >= len(dates):
        raise ValueError(f"Not enough price history after event date {event_date}")
    start_close = float(price_df.loc[price_df["date"] == dates.iloc[start_index], "close"].iloc[0])
    end_close = float(price_df.loc[price_df["date"] == dates.iloc[end_index], "close"].iloc[0])
    return end_close / start_close - 1.0


def build_targets(events_df: pd.DataFrame, stock_prices: pd.DataFrame, market_prices: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Add stock_return_5d, market_return_5d, abnormal_return_5d, and target_up."""
    rows = []
    for _, event in events_df.iterrows():
        stock_return = compute_forward_return(stock_prices, event["earnings_date"], horizon=horizon)
        market_return = compute_forward_return(market_prices, event["earnings_date"], horizon=horizon)
        abnormal_return = stock_return - market_return
        rows.append(
            {
                "event_id": event["event_id"],
                "ticker": event["ticker"],
                "earnings_date": event["earnings_date"],
                f"stock_return_{horizon}d": stock_return,
                f"market_return_{horizon}d": market_return,
                f"abnormal_return_{horizon}d": abnormal_return,
                "target_up": int(abnormal_return > 0),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_data_dirs()
    events = load_events(EVENTS_CSV)
    ticker = str(events["ticker"].iloc[0])
    targets = build_targets(events, load_price_data(PRICES_DIR / f"{ticker}.csv"), load_price_data(PRICES_DIR / "SPY.csv"))
    Path(FEATURES_DIR).mkdir(parents=True, exist_ok=True)
    targets.to_csv(TARGETS_CSV, index=False)
    print(f"Saved targets to {TARGETS_CSV}")


if __name__ == "__main__":
    main()
