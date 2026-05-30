from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import run_event_backtest, summarize_backtest
from src.config import (
    LIGHTGBM_MODEL_PATH,
    MODELING_DATASET_CSV,
    PREDICTIONS_CSV,
    PREDICTIONS_DIR,
    QUESTIONS,
)
from src.model_evaluation import evaluate_classifier
from src.model_training import get_feature_columns, load_model_bundle
from src.rag_pipeline import answer_question
from src.vector_store import get_available_documents


st.set_page_config(page_title="Earnings RAG Quant Dashboard", layout="wide")


def _read_csv_if_exists(path: Path, **kwargs) -> pd.DataFrame | None:
    if not path.exists():
        st.warning(f"Missing file: {path}")
        return None
    return pd.read_csv(path, **kwargs)


def _select_or_placeholder(label: str, values: list, default=None):
    options = sorted({value for value in values if pd.notna(value)})
    if default is not None and default not in options:
        options.insert(0, default)
    if not options:
        st.sidebar.warning(f"No values available for {label}.")
        return default
    return st.sidebar.selectbox(label, options)


def render_rag_page() -> None:
    st.header("RAG Analysis")
    docs = get_available_documents()

    if not docs:
        st.info(
            "No vector metadata found yet. Build the vector store before running RAG analysis."
        )
        return

    docs_df = pd.DataFrame(docs)
    ticker = _select_or_placeholder("Ticker", docs_df["ticker"].tolist(), "MSFT")
    filtered = docs_df[docs_df["ticker"] == ticker]

    fiscal_year = _select_or_placeholder("Fiscal Year", filtered["fiscal_year"].tolist())
    filtered = filtered[filtered["fiscal_year"] == fiscal_year]

    quarter = _select_or_placeholder("Quarter", filtered["quarter"].tolist())
    filtered = filtered[filtered["quarter"] == quarter]

    document_type = _select_or_placeholder(
        "Document Type", filtered["document_type"].tolist(), "10Q"
    )
    question = st.sidebar.selectbox("Question", QUESTIONS)

    filters = {
        "ticker": ticker,
        "fiscal_year": int(fiscal_year) if fiscal_year is not None else fiscal_year,
        "quarter": quarter,
        "document_type": document_type,
    }

    st.subheader("Selection")
    st.json({**filters, "question": question})

    if st.button("Run RAG Analysis"):
        with st.spinner("Running local RAG analysis..."):
            answer = answer_question(question, filters=filters)

        st.subheader("Answer")
        st.write(answer.answer)

        c1, c2 = st.columns(2)
        c1.metric("Signal", answer.signal)
        c2.metric("Confidence", f"{answer.confidence:.2f}")

        st.subheader("Evidence")
        if answer.evidence:
            for item in answer.evidence:
                with st.expander(f"{item.doc_id} page {item.page}"):
                    st.write(item.text)
        else:
            st.write("No evidence returned.")

        st.subheader("Retrieved Chunks")
        if answer.retrieved_chunks:
            for chunk in answer.retrieved_chunks:
                title = (
                    f"{chunk.get('chunk_id')} | page {chunk.get('page')} | "
                    f"score {chunk.get('similarity_score', 0):.3f}"
                )
                with st.expander(title):
                    st.write(chunk.get("text", ""))
        else:
            st.write("No chunks returned.")

        st.subheader("Limitations")
        st.write(answer.limitations)


def render_dataset_page() -> None:
    st.header("Feature Dataset")
    st.warning(
        "This MVP validates the end-to-end pipeline. The current sample size is too small "
        "to draw statistically reliable conclusions."
    )

    df = _read_csv_if_exists(MODELING_DATASET_CSV, parse_dates=["earnings_date"])
    if df is None:
        return

    feature_cols = get_feature_columns(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Events", len(df))
    c2.metric("Model Features", len(feature_cols))
    c3.metric("Target Up Rate", f"{df['target_up'].mean():.2%}")

    left, right = st.columns(2)
    with left:
        st.subheader("Target Distribution")
        target_counts = df["target_up"].value_counts().rename_axis("target_up").reset_index(
            name="count"
        )
        st.plotly_chart(px.bar(target_counts, x="target_up", y="count"), use_container_width=True)

    with right:
        st.subheader("Abnormal Return Distribution")
        st.plotly_chart(px.histogram(df, x="abnormal_return_5d"), use_container_width=True)

    st.subheader("Modeling Dataset")
    st.dataframe(df, use_container_width=True)


def render_prediction_page() -> None:
    st.header("Prediction")
    st.warning(
        "Model metrics here are pipeline checks, not reliable investment evidence yet."
    )

    predictions = _read_csv_if_exists(PREDICTIONS_CSV, parse_dates=["earnings_date"])
    dataset = _read_csv_if_exists(MODELING_DATASET_CSV, parse_dates=["earnings_date"])
    if predictions is None or dataset is None:
        return

    model_names = sorted(predictions["model_name"].unique())
    model_name = st.selectbox("Model", model_names)
    model_predictions = predictions[predictions["model_name"] == model_name]

    st.subheader("Predictions")
    st.dataframe(model_predictions, use_container_width=True)

    model_path = LIGHTGBM_MODEL_PATH if model_name == "LightGBM" else PROJECT_ROOT / "models" / "logistic_regression.joblib"
    if model_path.exists():
        bundle = load_model_bundle(model_path)
        model = bundle["model"]
        feature_cols = bundle.get("feature_cols") or get_feature_columns(dataset)
        eval_df = dataset.sort_values("earnings_date").tail(len(model_predictions))
        metrics = evaluate_classifier(model, eval_df, feature_cols)

        st.subheader("Metrics")
        st.json(metrics)

        matrix = metrics.get("confusion_matrix")
        if matrix:
            cm_df = pd.DataFrame(matrix, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
            st.subheader("Confusion Matrix")
            st.dataframe(cm_df)

        if model_name == "LightGBM" and hasattr(model, "feature_importances_"):
            importances = pd.DataFrame(
                {
                    "feature": feature_cols,
                    "importance": model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)
            st.subheader("Feature Importance")
            st.plotly_chart(
                px.bar(importances.head(20), x="importance", y="feature", orientation="h"),
                use_container_width=True,
            )

    metrics_path = PREDICTIONS_DIR / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as fp:
            st.subheader("Saved Metrics")
            st.json(json.load(fp))


def render_backtest_page() -> None:
    st.header("Backtest")
    st.warning(
        "This backtest is intentionally simple and only validates wiring across the pipeline."
    )

    predictions = _read_csv_if_exists(PREDICTIONS_CSV, parse_dates=["earnings_date"])
    if predictions is None:
        return

    model_name = st.selectbox("Model", sorted(predictions["model_name"].unique()))
    threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.6, 0.05)
    transaction_cost = st.number_input(
        "Transaction Cost", min_value=0.0, max_value=0.05, value=0.001, step=0.0005
    )

    model_predictions = predictions[predictions["model_name"] == model_name].sort_values(
        "earnings_date"
    )
    backtest_df = run_event_backtest(model_predictions, threshold, transaction_cost)
    summary = summarize_backtest(backtest_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Trades", summary["number_of_trades"])
    c2.metric("Average Trade Return", f"{summary['average_trade_return']:.2%}")
    c3.metric("Win Rate", f"{summary['win_rate']:.2%}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Cumulative Return", f"{summary['cumulative_return']:.2%}")
    c5.metric("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}")
    c6.metric("Max Drawdown", f"{summary['max_drawdown']:.2%}")

    st.subheader("Cumulative Return")
    st.plotly_chart(
        px.line(backtest_df, x="earnings_date", y="cumulative_return", markers=True),
        use_container_width=True,
    )

    st.subheader("Event Returns")
    st.dataframe(backtest_df, use_container_width=True)


def main() -> None:
    st.title("Earnings RAG Quant Dashboard")
    page = st.sidebar.radio(
        "Page",
        ["RAG Analysis", "Feature Dataset", "Prediction", "Backtest"],
    )

    if page == "RAG Analysis":
        render_rag_page()
    elif page == "Feature Dataset":
        render_dataset_page()
    elif page == "Prediction":
        render_prediction_page()
    else:
        render_backtest_page()


if __name__ == "__main__":
    main()
