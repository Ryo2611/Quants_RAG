from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import LIGHTGBM_MODEL_PATH, LOGISTIC_MODEL_PATH, MODELING_DATASET_CSV, ensure_data_dirs

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


def split_train_test_by_time(df: pd.DataFrame, date_col: str = "earnings_date", train_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort by date and split into train/test."""
    ordered = df.copy()
    ordered[date_col] = pd.to_datetime(ordered[date_col])
    ordered = ordered.sort_values(date_col).reset_index(drop=True)
    split_index = max(1, min(len(ordered) - 1, int(len(ordered) * train_ratio))) if len(ordered) > 1 else len(ordered)
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model feature columns excluding identifiers, targets, returns, and text columns."""
    excluded = {
        "event_id", "ticker", "company_name", "fiscal_year", "quarter", "document_type", "doc_id",
        "earnings_date", "filing_date", "target_up", "stock_return_5d", "market_return_5d", "abnormal_return_5d",
    }
    return [
        column for column in df.columns
        if column not in excluded
        and not column.endswith(("_answer", "_evidence"))
        and pd.api.types.is_numeric_dtype(df[column])
    ]


def _needs_dummy(train_df: pd.DataFrame, target_col: str) -> bool:
    return len(train_df) < 2 or train_df[target_col].nunique() < 2


def train_logistic_regression(train_df: pd.DataFrame, feature_cols: list[str], target_col: str = "target_up"):
    """Train baseline logistic regression model."""
    if _needs_dummy(train_df, target_col) or not feature_cols:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(train_df[feature_cols] if feature_cols else [[0]] * len(train_df), train_df[target_col])
        return model
    model = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    model.fit(train_df[feature_cols], train_df[target_col])
    return model


def train_lightgbm(train_df: pd.DataFrame, feature_cols: list[str], target_col: str = "target_up"):
    """Train LightGBM classifier."""
    if LGBMClassifier is None or _needs_dummy(train_df, target_col) or not feature_cols:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(train_df[feature_cols] if feature_cols else [[0]] * len(train_df), train_df[target_col])
        return model
    model = LGBMClassifier(n_estimators=50, learning_rate=0.05, max_depth=3, min_child_samples=1, random_state=42, verbose=-1)
    model.fit(train_df[feature_cols], train_df[target_col])
    return model


def save_model(model, path: str | Path, feature_cols: list[str] | None = None) -> None:
    """Save trained model using joblib."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols or []}, output)


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle
    return {"model": bundle, "feature_cols": []}


def main() -> None:
    ensure_data_dirs()
    df = pd.read_csv(MODELING_DATASET_CSV, parse_dates=["earnings_date"])
    train_df, _ = split_train_test_by_time(df)
    feature_cols = get_feature_columns(df)
    save_model(train_logistic_regression(train_df, feature_cols), LOGISTIC_MODEL_PATH, feature_cols)
    save_model(train_lightgbm(train_df, feature_cols), LIGHTGBM_MODEL_PATH, feature_cols)
    print(f"Saved models to {LOGISTIC_MODEL_PATH} and {LIGHTGBM_MODEL_PATH}")


if __name__ == "__main__":
    main()
