import pandas as pd

from src.target_builder import build_targets, compute_forward_return


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-29",
                    "2025-01-30",
                    "2025-01-31",
                    "2025-02-03",
                    "2025-02-04",
                    "2025-02-05",
                ]
            ),
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 106.0],
        }
    )


def test_compute_forward_return_uses_t_plus_1_to_t_plus_horizon() -> None:
    result = compute_forward_return(_prices(), "2025-01-29", horizon=5)

    assert result == 106.0 / 101.0 - 1.0


def test_build_targets_creates_abnormal_return_and_label() -> None:
    events = pd.DataFrame(
        [
            {
                "event_id": "MSFT_2025_Q2",
                "ticker": "MSFT",
                "earnings_date": pd.Timestamp("2025-01-29"),
            }
        ]
    )
    stock = _prices()
    market = _prices().assign(close=[100.0, 100.0, 100.0, 100.0, 100.0, 100.0])

    targets = build_targets(events, stock, market, horizon=5)

    assert targets.loc[0, "abnormal_return_5d"] > 0
    assert targets.loc[0, "target_up"] == 1
