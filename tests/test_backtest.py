import pandas as pd
import pytest

from src.backtest import run_event_backtest, summarize_backtest


def test_run_event_backtest_trades_above_threshold() -> None:
    predictions = pd.DataFrame(
        {
            "pred_proba_up": [0.7, 0.4],
            "stock_return_5d": [0.03, -0.02],
        }
    )

    results = run_event_backtest(predictions, threshold=0.6, transaction_cost=0.001)

    assert results["trade"].tolist() == [True, False]
    assert results["strategy_return"].tolist() == [pytest.approx(0.029), 0.0]


def test_summarize_backtest_handles_no_trades() -> None:
    results = run_event_backtest(
        pd.DataFrame({"pred_proba_up": [0.1], "stock_return_5d": [0.02]}),
        threshold=0.6,
    )

    summary = summarize_backtest(results)

    assert summary["number_of_trades"] == 0
    assert summary["cumulative_return"] == 0.0
