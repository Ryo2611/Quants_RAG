# Quants_RAG
# Earnings RAG Quant Dashboard

This project builds a RAG-based financial document analysis system that extracts structured signals from earnings reports and prepares them for post-earnings return prediction.

## Scope

- Load earnings-related PDF documents
- Extract page-level text
- Chunk documents
- Build a vector database
- Retrieve relevant passages
- Generate evidence-based financial summaries using an LLM
- Display results in a Streamlit web app
- Build RAG/LLM event features
- Download stock and benchmark prices
- Create 5-day post-earnings abnormal return targets
- Train baseline Logistic Regression and LightGBM classifiers
- Run a simple event-driven backtest

## Project Structure

```text
earnings-rag-quant-dashboard/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw_pdfs/
│   ├── processed_text/
│   ├── metadata/
│   ├── vector_db/
│   ├── events/
│   ├── prices/
│   ├── features/
│   ├── predictions/
│   └── backtest/
├── models/
├── notebooks/
├── src/
│   ├── config.py
│   ├── pdf_loader.py
│   ├── chunking.py
│   ├── vector_store.py
│   ├── rag_pipeline.py
│   ├── price_loader.py
│   ├── target_builder.py
│   ├── llm_feature_builder.py
│   ├── price_feature_builder.py
│   ├── dataset_builder.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── backtest.py
│   └── schemas.py
├── tests/
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

### 1. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Install Ollama (Local LLM)

Download and install Ollama from https://ollama.com, then pull a model:

```bash
ollama pull llama3.2
```

Ollama runs as a local server at `http://localhost:11434`. No API key is needed.

### 3. Configuration (`.env`)

All settings are optional. Defaults work out of the box:

```text
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.2
EMBEDDING_BACKEND=hash
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## PDF Naming

Place PDF files in `data/raw_pdfs/` using this format:

```text
{ticker}_{fiscal_year}_{quarter}_{document_type}.pdf
```

Example:

```text
MSFT_2024_Q1_10Q.pdf
```

## Run The RAG Pipeline

```bash
python -m src.pdf_loader
python -m src.chunking
python -m src.vector_store
streamlit run app/streamlit_app.py
```

This project runs entirely locally with no paid API keys. By default, embeddings use a deterministic local hash backend so the project runs without downloading a Hugging Face model. LLM answers are generated with Ollama. PDF extraction and chunking have no additional dependencies.

For better retrieval quality, set `EMBEDDING_BACKEND=sentence-transformers` after the `all-MiniLM-L6-v2` model is available locally. If SentenceTransformers fails to load, the vector store falls back to local hash embeddings. When the Ollama LLM call fails, for example if Ollama is not running, the app returns an extractive fallback answer instead of crashing.

If Ollama is installed but not running, start it before using RAG generation:

```bash
ollama serve
```

`python -m src.vector_store` rebuilds the Chroma collection from `data/processed_text/chunks.jsonl`, so it is safe to rerun after adding, removing, or renaming PDFs.

The generated files are:

- `data/processed_text/pages.jsonl`
- `data/processed_text/chunks.jsonl`
- `data/vector_db/`

## Run The Phase 2 Prediction Pipeline

First make sure `data/events/earnings_events.csv` exists and that each `doc_id` matches a PDF filename without `.pdf`.

```bash
python -m src.price_loader
python -m src.target_builder
python -m src.llm_feature_builder
python -m src.price_feature_builder
python -m src.dataset_builder
python -m src.model_training
python -m src.model_evaluation
python -m src.backtest
streamlit run app/streamlit_app.py
```

Generated Phase 2 artifacts:

- `data/prices/MSFT.csv`
- `data/prices/SPY.csv`
- `data/features/targets.csv`
- `data/features/llm_features.csv`
- `data/features/price_features.csv`
- `data/features/modeling_dataset.csv`
- `models/logistic_regression.joblib`
- `models/lightgbm_model.joblib`
- `data/predictions/predictions.csv`
- `data/backtest/backtest_results.csv`

## Streamlit Pages

- `RAG Analysis`
- `Feature Dataset`
- `Prediction`
- `Backtest`

## Fixed MVP Questions

- What were the main drivers of revenue growth?
- Did operating income improve or deteriorate?
- What risks or uncertainties were mentioned?
- Was the management tone positive, neutral, or negative?
- What factors could affect the stock price after earnings?

## Multiple PDFs And Quarters

Add any number of PDFs to `data/raw_pdfs/` using the filename convention above, then rerun the pipeline. The Streamlit app reads indexed document metadata and lets you select the exact ticker/year/quarter/document combination.

For Phase 2, also add matching rows to `data/events/earnings_events.csv`.

## Modeling Caveat

This MVP is designed to validate the end-to-end pipeline.
The current sample size is too small to draw statistically reliable conclusions.
The next step is to expand the dataset to multiple companies and multiple fiscal years.

The implementation avoids look-ahead bias by using only earnings document text and pre-event price features as model inputs. Post-event returns are used only as targets and for backtest evaluation.

## Future Work

- Compare models with and without RAG features
- Add more companies and fiscal years
- Add SEC filing and earnings-date auto-discovery
- Add richer financial statement features
