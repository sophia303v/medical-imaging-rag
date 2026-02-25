# Medical Imaging Multimodal RAG

A Retrieval-Augmented Generation system for radiology reports. Ask questions about chest X-ray findings — optionally upload an image — and get AI-generated answers grounded in a knowledge base of radiology reports, with source citations.

## Features

### RAG Pipeline
- **Multimodal retrieval** — text queries + optional chest X-ray image input (Gemini Vision converts images to text for search)
- **Section-aware chunking** — splits radiology reports by natural boundaries (indication / findings / impression) instead of fixed token windows
- **Dual embedding backend** — TF-IDF for offline development, Gemini `text-embedding-004` for production
- **ChromaDB vector store** — persistent, embedded database with cosine similarity search
- **Grounded generation** — Gemini 2.0 Flash generates answers with source citations and medical disclaimers
- **Local fallback** — works without API key by returning raw retrieved documents

### Evaluation System
- **6 retrieval metrics** — Context Precision, Context Recall, MRR (Mean Reciprocal Rank), nDCG (Normalized Discounted Cumulative Gain), Faithfulness, Answer Relevancy
- **3-criteria LLM Judge** — Medical Appropriateness, Citation Accuracy, Answer Completeness
- **Multiple benchmark datasets** — custom medical QA (40 pairs), SQuAD v2 (2000 pairs), SciFact (300 queries)
- **Experiment management** — YAML configs, auto-named experiment folders, config snapshots
- **Interactive HTML report** — radar chart, grouped bar chart, metric descriptions, per-question detail table (Plotly)
- **Cross-experiment comparison** — overlaid radar charts, delta tables, config diff
- **Graceful degradation** — retrieval metrics work without API key; LLM metrics are optional

### Web Interface
- **Gradio UI** on port 7860 — image upload, text input, example questions, source display with relevance scores

## Experiment Results

All experiments use `all-MiniLM-L12-v2` (sentence-transformers) for embedding and ChromaDB for retrieval.

| Experiment | Dataset | K | Questions | Mode | Precision | Recall | MRR | nDCG | Faithfulness | Relevancy |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | openi_synthetic | 3 | 40 | Retrieval | 0.321 | 0.642 | — | — | — | — |
| topk5 | openi_synthetic | 5 | 40 | Retrieval | 0.216 | 0.694 | — | — | — | — |
| scifact_baseline | scifact | 3 | 300 | Retrieval | 0.223 | 0.614 | 0.550 | 0.559 | — | — |
| scifact_topk1 | scifact | 1 | 300 | Retrieval | 0.483 | 0.460 | 0.483 | 0.483 | — | — |
| squad_v2_baseline | squad_v2 | 3 | 2000 | Retrieval | 0.246 | 0.737 | 0.644 | 0.668 | — | — |
| squad_v2_full_20 | squad_v2 | 3 | 20 | Full | 0.150 | 0.450 | 0.317 | 0.351 | 0.812 | 0.510 |
| squad_v2_topk1 | squad_v2 | 1 | 2000 | Retrieval | 0.565 | 0.565 | 0.565 | 0.565 | — | — |

**Key observations:**
- **Precision vs Recall trade-off**: K=1 gives higher precision but lower recall; K=3 retrieves more relevant docs but dilutes precision
- **MRR ~0.6 on SQuAD**: the first relevant document is typically at rank 1-2
- **Faithfulness 0.812**: generated answers are mostly grounded in retrieved context (not hallucinating)
- **Relevancy 0.510**: moderate — SQuAD questions are general knowledge, not medical, so the medical system prompt adds noise

Each experiment generates an interactive HTML report in `experiments/{date}_{name}/eval_report.html`.

## Project Structure

```
medical-imaging-rag/
├── app.py                    # Gradio web interface
├── ingest.py                 # CLI data ingestion script
├── run_eval.py               # CLI evaluation entry point
├── config.py                 # Centralized configuration
├── requirements.txt
├── ARCHITECTURE.md           # Detailed architecture documentation
│
├── src/
│   ├── data_loader.py        # Load reports from HuggingFace or local JSON
│   ├── chunking.py           # Section-based text chunking
│   ├── embedding.py          # ST / TF-IDF / Gemini embedding backends
│   ├── vector_store.py       # ChromaDB indexing & search
│   ├── retriever.py          # Multimodal retrieval (text + image)
│   ├── generator.py          # Gemini / Ollama answer generation with citations
│   ├── rag_pipeline.py       # Orchestrator (ingest + query)
│   └── evaluation/
│       ├── metrics.py        # Retrieval metrics + LLM-based metrics
│       ├── llm_judge.py      # 3-criteria LLM judge
│       ├── runner.py         # Evaluation orchestrator (batch embed + concurrent eval)
│       └── visualization.py  # Plotly HTML report + cross-experiment comparison
│
├── scripts/
│   └── download_datasets.py  # Download SQuAD v2 / SciFact / RadQA benchmarks
│
├── experiments/              # Auto-generated experiment results
│   └── {date}_{name}/
│       ├── config.yaml       # Parameter snapshot
│       ├── eval_results.json # Full scores + generated answers
│       └── eval_report.html  # Interactive HTML report
│
└── data/
    ├── openi_synthetic/      # 20 medical reports + 40 QA pairs
    ├── squad_v2/             # SQuAD v2 benchmark (1204 passages, 2000 QA)
    ├── scifact/              # SciFact benchmark (5183 docs, 300 queries)
    └── chroma_db/            # Vector store (rebuilt per dataset)
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up API key (optional)

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
EMBEDDING_BACKEND=gemini    # or "local" for offline TF-IDF (default)
```

Without an API key the system still works — it uses TF-IDF embeddings and returns retrieved documents directly instead of generating LLM answers.

### 3. Ingest data

```bash
python ingest.py
```

This loads the 20 sample radiology reports, chunks them by section, generates embeddings, and stores them in ChromaDB. Only needs to run once (data persists in `data/chroma_db/`).

### 4. Launch the web UI

```bash
python app.py
```

Open http://localhost:7860 in your browser. Type a question or click an example, optionally upload a chest X-ray image.

### 5. Run evaluation

```bash
# Full evaluation (requires GEMINI_API_KEY)
python run_eval.py

# Retrieval metrics only (no API key needed)
python run_eval.py --retrieval-only

# Run on benchmark datasets
python scripts/download_datasets.py squad_v2    # download first
python run_eval.py --dataset squad_v2 --retrieval-only --experiment-name squad_test

# Quick test with limited questions
python run_eval.py --dataset squad_v2 --max-samples 20 --experiment-name quick_test

# YAML config for reproducible experiments
python run_eval.py --config experiments/configs/example.yaml --experiment-name my_exp
```

Results are saved to `experiments/{date}_{name}/` with an interactive HTML report.

**CLI options:**

| Flag | Description |
|---|---|
| `--retrieval-only` | Skip LLM-based metrics (no API key needed) |
| `--dataset NAME` | Switch dataset (squad_v2, scifact, openi_synthetic) |
| `--max-samples N` | Limit to first N questions (useful for quick tests) |
| `--top-k N` | Override number of retrieved documents |
| `--experiment-name NAME` | Save results to `experiments/{date}_{name}/` |
| `--config PATH` | Load YAML config for parameter overrides |
| `--embedding-backend` | sentence-transformers, tfidf, or gemini |
| `--generation-backend` | gemini or ollama |
| `--qa PATH` | Custom golden QA file |
| `--output-dir PATH` | Output directory (default: `data/`) |
| `--quiet` | Suppress progress output |

## How It Works

```
User Question (+ optional X-ray)
        │
        ▼
┌─ Retriever ──────────────────────┐
│  1. Image → Gemini Vision desc.  │
│  2. Query → embedding            │
│  3. ChromaDB cosine search       │
└──────────────────────────────────┘
        │ top-5 relevant chunks
        ▼
┌─ Generator ──────────────────────┐
│  Gemini 2.0 Flash                │
│  + source citations              │
│  + medical disclaimer            │
└──────────────────────────────────┘
        │
        ▼
    Answer with references
```

## Evaluation Metrics

| Metric | Type | What it measures |
|---|---|---|
| Context Precision | Retrieval | % of retrieved docs that are relevant |
| Context Recall | Retrieval | % of ground truth docs that were retrieved |
| MRR (Reciprocal Rank) | Retrieval | 1 / position of first relevant doc |
| nDCG | Retrieval | Ranking quality, weighted by position |
| Faithfulness | Generation (LLM) | Is the answer grounded in retrieved context? |
| Answer Relevancy | Generation (LLM) | Does the answer address the question? |
| Medical Appropriateness | Judge (LLM) | Correct terminology, clinically accurate |
| Citation Accuracy | Judge (LLM) | Sources cited and match content |
| Answer Completeness | Judge (LLM) | Covers key points from ground truth |

## Data

The project uses 20 synthetic radiology reports (`data/sample_reports.json`) covering common chest X-ray findings: pneumonia, CHF, pneumothorax, COPD, lung masses, tuberculosis, ARDS, rib fractures, sarcoidosis, and more. For a larger dataset, the system can download the [Indiana University Chest X-ray (OpenI)](https://huggingface.co/datasets/ykumards/open-i) dataset from HuggingFace.

## Tech Stack

- **LLM**: Gemini 2.0 Flash (generation + evaluation)
- **Embeddings**: sentence-transformers `all-MiniLM-L12-v2` (default), Gemini `text-embedding-004`, or TF-IDF (offline)
- **Vector DB**: ChromaDB (embedded, persistent)
- **Web UI**: Gradio
- **Visualization**: Plotly
- **Data**: HuggingFace Datasets
