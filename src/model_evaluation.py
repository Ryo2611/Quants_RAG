from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    LIGHTGBM_MODEL_PATH,
    LOGISTIC_MODEL_PATH,
    MODELING_DATASET_CSV,
    PREDICTIONS_CSV,
    PREDICTIONS_DIR,
    ensure_data_dirs,
)
from src.model_training import get_feature_columns, load_model_bundle, split_train_test_by_time


def _predict_proba_up(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.shape[1] == 1:
            classes = getattr(model, "classes_", [0])
            return np.ones(len(x)) if int(classes[0]) == 1 else np.zeros(len(x))
        return proba[:, 1]
    return model.predict(x)


def evaluate_classifier(
    model,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_up",
) -> dict:
    """Evaluate classification and simple investment-oriented metrics."""
    if test_df.empty:
        return {"error": "No test rows available."}

    y_true = test_df[target_col].astype(int)
    proba = _predict_proba_up(model, test_df[feature_cols])
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
        "roc_auc": float(roc_auc_score(y_true, proba)) if y_true.nunique() > 1 else None,
    }

    up_returns = test_df.loc[pred == 1, "abnormal_return_5d"]
    down_returns = test_df.loc[pred == 0, "abnormal_return_5d"]
    metrics["average_return_when_predicted_up"] = (
        float(up_returns.mean()) if not up_returns.empty else None
    )
    metrics["average_return_when_predicted_down"] = (
        float(down_returns.mean()) if not down_returns.empty else None
    )
    metrics["hit_rate"] = float((pred == y_true).mean())
    return metrics


def create_predictions(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
) -> pd.DataFrame:
    """Create prediction rows for downstream evaluation and backtesting."""
    proba = _predict_proba_up(model, df[feature_cols])
    pred = (proba >= 0.5).astype(int)

    cols = [
        "event_id",
        "earnings_date",
        "ticker",
        "target_up",
        "abnormal_return_5d",
        "stock_return_5d",
    ]
    output = df[cols].copy()
    output["pred_proba_up"] = proba
    output["pred_label"] = pred
    output["model_name"] = model_name
    return output


def main() -> None:
    ensure_data_dirs()
    df = pd.read_csv(MODELING_DATASET_CSV, parse_dates=["earnings_date"])
    _, test_df = split_train_test_by_time(df)
    if test_df.empty:
        test_df = df.tail(1).copy()

    predictions = []
    metrics: dict[str, dict] = {}
    for name, path in [
        ("Logistic Regression", LOGISTIC_MODEL_PATH),
        ("LightGBM", LIGHTGBM_MODEL_PATH),
    ]:
        bundle = load_model_bundle(path)
        model = bundle["model"]
        feature_cols = bundle.get("feature_cols") or get_feature_columns(df)
        predictions.append(create_predictions(model, test_df, feature_cols, name))
        metrics[name] = evaluate_classifier(model, test_df, feature_cols)

    pred_df = pd.concat(predictions, ignore_index=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(PREDICTIONS_CSV, index=False)

    metrics_path = PREDICTIONS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved predictions to {PREDICTIONS_CSV}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
