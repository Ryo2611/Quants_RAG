from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BACKTEST_RESULTS_CSV, PREDICTIONS_CSV, ensure_data_dirs


def run_event_backtest(
    predictions_df: pd.DataFrame,
    threshold: float = 0.6,
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    """Run a simple long-only event backtest from prediction rows."""
    df = predictions_df.copy()
    df["trade"] = df["pred_proba_up"] >= threshold
    df["strategy_return"] = np.where(
        df["trade"],
        df["stock_return_5d"] - transaction_cost,
        0.0,
    )
    df["cumulative_return"] = (1.0 + df["strategy_return"]).cumprod() - 1.0
    return df


def summarize_backtest(backtest_df: pd.DataFrame) -> dict:
    """Calculate compact performance metrics for the event backtest."""
    trades = backtest_df[backtest_df["trade"]].copy()
    returns = trades["strategy_return"] if not trades.empty else pd.Series(dtype=float)

    cumulative_curve = (1.0 + backtest_df["strategy_return"]).cumprod()
    if cumulative_curve.empty:
        drawdown = pd.Series([0.0])
    else:
        drawdown = cumulative_curve / cumulative_curve.cummax() - 1.0

    return {
        "number_of_trades": int(len(trades)),
        "average_trade_return": float(returns.mean()) if not returns.empty else 0.0,
        "win_rate": float((returns > 0).mean()) if not returns.empty else 0.0,
        "cumulative_return": (
            float(cumulative_curve.iloc[-1] - 1.0) if not cumulative_curve.empty else 0.0
        ),
        "sharpe_ratio": (
            float(returns.mean() / returns.std(ddof=0) * np.sqrt(max(len(returns), 1)))
            if len(returns) > 1 and returns.std(ddof=0) != 0
            else 0.0
        ),
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
    }


def main() -> None:
    ensure_data_dirs()
    predictions = pd.read_csv(PREDICTIONS_CSV, parse_dates=["earnings_date"])
    model_name = (
        "LightGBM"
        if "LightGBM" in set(predictions["model_name"])
        else predictions["model_name"].iloc[0]
    )
    model_predictions = predictions[predictions["model_name"] == model_name].sort_values(
        "earnings_date"
    )
    results = run_event_backtest(model_predictions)
    BACKTEST_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(BACKTEST_RESULTS_CSV, index=False)

    print(f"Saved backtest results to {BACKTEST_RESULTS_CSV}")
    print(summarize_backtest(results))


if __name__ == "__main__":
    main()
