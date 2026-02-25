"""
Streamlit dashboard for Medical Imaging RAG.

Three tabs:
1. Eval Dashboard — interactive charts and tables from eval_results.json
2. RAG Query — live query interface with source inspection
3. Metric Guide — explains each evaluation metric
"""
import json
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METRIC_LABELS = {
    "context_precision": "Ctx Precision",
    "context_recall": "Ctx Recall",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Relevancy",
    "domain_appropriateness": "Domain",
    "citation_accuracy": "Citation",
    "answer_completeness": "Completeness",
}
METRIC_ORDER = list(METRIC_LABELS.keys())

COLORS = ["#0f3460", "#1a73e8", "#4CAF50", "#FF9800", "#9C27B0", "#E91E63", "#00BCD4"]

EXAMPLE_QUESTIONS = [
    "What are the most common findings in chest X-rays?",
    "Describe the typical appearance of pneumonia on chest radiograph.",
    "What does cardiomegaly indicate and how is it identified?",
    "What are the signs of pleural effusion on X-ray?",
    "How can you differentiate between consolidation and atelectasis?",
]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Medical Imaging RAG",
    page_icon="\U0001fa7a",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fmt(score: float) -> str:
    return "N/A" if score < 0 else f"{score:.3f}"


def _score_color(score: float) -> str:
    if score < 0:
        return "gray"
    if score >= 0.7:
        return "green"
    if score >= 0.4:
        return "orange"
    return "red"


def _group_mean(values: list[float]) -> float:
    valid = [v for v in values if v >= 0]
    return sum(valid) / len(valid) if valid else -1.0


@st.cache_data
def load_eval_results(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tab 1: Eval Dashboard
# ---------------------------------------------------------------------------
def render_eval_dashboard():
    data = load_eval_results(str(config.EVAL_RESULTS_PATH))
    if data is None:
        st.warning(
            f"No eval results found at `{config.EVAL_RESULTS_PATH}`. "
            "Run `python run_eval.py` first."
        )
        return

    agg = data["aggregate_scores"]
    by_cat = data["scores_by_category"]
    results = data["results"]

    # --- Filters ---
    categories = sorted({r["category"] for r in results})
    difficulties = sorted({r["difficulty"] for r in results})

    with st.sidebar:
        st.header("Filters")
        sel_categories = st.multiselect("Category", categories, default=categories)
        sel_difficulties = st.multiselect("Difficulty", difficulties, default=difficulties)
        sort_metric = st.selectbox(
            "Sort questions by",
            ["question_id"] + [METRIC_LABELS[m] for m in METRIC_ORDER],
        )

    # Apply filters
    filtered = [
        r for r in results
        if r["category"] in sel_categories and r["difficulty"] in sel_difficulties
    ]

    # Recompute aggregates on filtered data
    def _filtered_agg(items):
        out = {}
        for m in METRIC_ORDER:
            vals = [r["scores"][m] for r in items if r["scores"][m] >= 0]
            if vals:
                out[m] = {
                    "mean": sum(vals) / len(vals),
                    "min": min(vals),
                    "max": max(vals),
                    "count": len(vals),
                }
            else:
                out[m] = {"mean": -1.0, "min": -1.0, "max": -1.0, "count": 0}
        return out

    f_agg = _filtered_agg(filtered)

    # --- Stat boxes ---
    all_means = [f_agg[m]["mean"] for m in METRIC_ORDER if f_agg[m]["mean"] >= 0]
    overall = sum(all_means) / len(all_means) if all_means else 0

    ret_means = [f_agg[m]["mean"] for m in ["context_precision", "context_recall"] if f_agg[m]["mean"] >= 0]
    retrieval_avg = sum(ret_means) / len(ret_means) if ret_means else 0

    gen_means = [f_agg[m]["mean"] for m in ["faithfulness", "answer_relevancy"] if f_agg[m]["mean"] >= 0]
    gen_avg = sum(gen_means) / len(gen_means) if gen_means else 0

    judge_means = [f_agg[m]["mean"] for m in ["domain_appropriateness", "citation_accuracy", "answer_completeness"] if f_agg[m]["mean"] >= 0]
    judge_avg = sum(judge_means) / len(judge_means) if judge_means else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Score", f"{overall:.3f}", help="Mean across all 7 metrics")
    c2.metric("Retrieval Quality", f"{retrieval_avg:.3f}", help="Ctx Precision + Recall")
    c3.metric("Generation Quality", f"{gen_avg:.3f}", help="Faithfulness + Relevancy")
    c4.metric(
        "Domain Judge",
        f"{judge_avg:.3f}" if judge_avg > 0 else "N/A",
        delta=f"{len(filtered)} questions",
        delta_color="off",
        help="Domain + Citation + Completeness",
    )

    st.divider()

    # --- Charts ---
    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.subheader("Overall Metrics")
        labels = [METRIC_LABELS[m] for m in METRIC_ORDER]
        values = [max(0, f_agg[m]["mean"]) for m in METRIC_ORDER]
        radar = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            fillcolor="rgba(15,52,96,0.15)",
            line=dict(color="#0f3460", width=2),
            marker=dict(size=6, color="#0f3460"),
        ))
        radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickvals=[0.2, 0.4, 0.6, 0.8, 1.0])),
            margin=dict(t=30, b=30, l=60, r=60),
            height=400,
            showlegend=False,
        )
        st.plotly_chart(radar, use_container_width=True)

    with chart_right:
        st.subheader("Scores by Category")
        # Recompute by-category from filtered data
        f_by_cat: dict[str, list] = {}
        for r in filtered:
            f_by_cat.setdefault(r["category"], []).append(r)
        cat_names = sorted(f_by_cat.keys())
        bar = go.Figure()
        for i, m in enumerate(METRIC_ORDER):
            bar.add_trace(go.Bar(
                x=cat_names,
                y=[_group_mean([r["scores"][m] for r in f_by_cat[c]]) for c in cat_names],
                name=METRIC_LABELS[m],
                marker_color=COLORS[i],
            ))
        bar.update_layout(
            barmode="group",
            yaxis=dict(range=[0, 1], title="Score"),
            xaxis=dict(title="Category"),
            margin=dict(t=30, b=60, l=50, r=20),
            height=400,
            legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center", font=dict(size=10)),
        )
        st.plotly_chart(bar, use_container_width=True)

    # --- Summary table ---
    st.subheader("Metric Summary")
    summary_rows = []
    for m in METRIC_ORDER:
        s = f_agg[m]
        summary_rows.append({
            "Metric": METRIC_LABELS[m],
            "Mean": _fmt(s["mean"]),
            "Min": _fmt(s["min"]),
            "Max": _fmt(s["max"]),
            "Count": s["count"],
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # --- Per-question detail ---
    st.subheader("Per-Question Results")

    # Sort
    if sort_metric == "question_id":
        sorted_items = sorted(filtered, key=lambda r: r["question_id"])
    else:
        metric_key = [k for k, v in METRIC_LABELS.items() if v == sort_metric][0]
        sorted_items = sorted(filtered, key=lambda r: r["scores"].get(metric_key, -1), reverse=True)

    for r in sorted_items:
        scores_preview = "  |  ".join(
            f"{METRIC_LABELS[m]}: {_fmt(r['scores'][m])}"
            for m in METRIC_ORDER
        )
        with st.expander(f"**{r['question_id']}** [{r['category']}/{r['difficulty']}]  —  {r['question'][:80]}"):
            st.caption(scores_preview)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**RAG Answer**")
                st.markdown(r["answer"])
            with col_b:
                st.markdown("**Ground Truth**")
                st.markdown(r["ground_truth"])

            st.markdown(f"**Retrieved UIDs:** `{'`, `'.join(r['retrieved_uids'])}`")
            st.markdown(f"**Relevant UIDs:** `{'`, `'.join(r['relevant_uids'])}`")

            # Score badges
            score_cols = st.columns(len(METRIC_ORDER))
            for j, m in enumerate(METRIC_ORDER):
                v = r["scores"][m]
                score_cols[j].markdown(
                    f"<div style='text-align:center;padding:4px;border-radius:4px;"
                    f"background:{_score_color(v)};color:white;font-size:0.85em'>"
                    f"{METRIC_LABELS[m]}<br><b>{_fmt(v)}</b></div>",
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Tab 2: RAG Query
# ---------------------------------------------------------------------------
@st.cache_resource
def get_rag_pipeline():
    from src.rag_pipeline import MedicalImagingRAG
    rag = MedicalImagingRAG()
    if not rag._is_ingested:
        rag.ingest()
    return rag


def render_rag_query():
    st.subheader("Ask a question about medical imaging")

    # Example buttons
    st.caption("Try an example:")
    btn_cols = st.columns(len(EXAMPLE_QUESTIONS))
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if btn_cols[i].button(q[:30] + "...", key=f"ex_{i}", use_container_width=True):
            st.session_state["rag_query"] = q

    question = st.text_area(
        "Your question",
        value=st.session_state.get("rag_query", ""),
        height=100,
        placeholder="e.g., What does cardiomegaly look like on a chest X-ray?",
    )

    submitted = st.button("Submit", type="primary", use_container_width=True)

    if submitted and question.strip():
        with st.spinner("Retrieving and generating..."):
            try:
                rag = get_rag_pipeline()
                result = rag.query(question.strip())
            except Exception as e:
                st.error(f"Query failed: {e}")
                return

        # Answer
        st.markdown("### Answer")
        st.markdown(result.answer)

        # Sources
        with st.expander("Sources", expanded=True):
            for i, (doc, meta, dist) in enumerate(zip(
                result.retrieval.documents,
                result.retrieval.metadatas,
                result.retrieval.distances,
            )):
                relevance = max(0.0, (1 - dist))
                uid = meta.get("uid", "unknown")
                section = meta.get("section", "unknown")
                st.markdown(f"**[Source {i+1}]** Report `{uid}` ({section})")
                st.progress(relevance, text=f"Relevance: {relevance * 100:.1f}%")
                st.caption(doc[:300] + ("..." if len(doc) > 300 else ""))

            if result.retrieval.image_description:
                st.markdown("**Image Analysis**")
                st.info(result.retrieval.image_description)

        # Debug
        with st.expander("Debug: full prompt"):
            st.code(result.prompt_used, language="text")


# ---------------------------------------------------------------------------
# Tab 3: Metric Guide
# ---------------------------------------------------------------------------
METRIC_GUIDE = {
    "context_precision": {
        "name": "Context Precision",
        "group": "Retrieval",
        "requires_llm": False,
        "formula": "Precision = |retrieved ∩ relevant| / |retrieved|",
        "description": (
            "What fraction of retrieved documents are actually relevant to the question? "
            "A high score means the retriever is not pulling in irrelevant noise."
        ),
        "scoring": {
            "1.0": "Every retrieved document is relevant",
            "0.5": "Half the retrieved documents are relevant",
            "0.0": "None of the retrieved documents are relevant",
        },
        "example": (
            "If the system retrieves 5 documents and 2 of them are actually relevant "
            "to the question, precision = 2/5 = 0.40."
        ),
    },
    "context_recall": {
        "name": "Context Recall",
        "group": "Retrieval",
        "requires_llm": False,
        "formula": "Recall = |retrieved ∩ relevant| / |relevant|",
        "description": (
            "What fraction of ground-truth relevant documents were successfully retrieved? "
            "A high score means the retriever finds most of the documents it should."
        ),
        "scoring": {
            "1.0": "All relevant documents were retrieved",
            "0.5": "Half of the relevant documents were retrieved",
            "0.0": "None of the relevant documents were retrieved",
        },
        "example": (
            "If 3 documents are marked as relevant in the golden QA set and the system "
            "retrieves 2 of them, recall = 2/3 = 0.67."
        ),
    },
    "faithfulness": {
        "name": "Faithfulness",
        "group": "Generation",
        "requires_llm": True,
        "formula": "LLM judges whether each claim is supported by the context",
        "description": (
            "Is the generated answer grounded in the retrieved context? "
            "A faithful answer does not hallucinate facts beyond what the source documents contain."
        ),
        "scoring": {
            "1.0": "Every claim in the answer is supported by the context",
            "0.7 - 0.9": "Most claims supported, minor unsupported details",
            "0.4 - 0.6": "Mix of supported and unsupported claims",
            "0.0 - 0.3": "Mostly unsupported or contradicts the context",
        },
        "example": (
            "If the context mentions 'right lower lobe opacity' and the answer says "
            "'right lower lobe opacity consistent with pneumonia', the pneumonia inference "
            "may lower faithfulness if not stated in the source."
        ),
    },
    "answer_relevancy": {
        "name": "Answer Relevancy",
        "group": "Generation",
        "requires_llm": True,
        "formula": "LLM judges whether the answer addresses the question",
        "description": (
            "Does the answer actually address what was asked? "
            "An irrelevant answer might be factually correct but off-topic."
        ),
        "scoring": {
            "1.0": "Directly and completely addresses the question",
            "0.7 - 0.9": "Mostly on-topic with minor tangents",
            "0.4 - 0.6": "Partially addresses the question",
            "0.0 - 0.3": "Mostly or entirely off-topic",
        },
        "example": (
            "Q: 'What are the signs of pleural effusion?' — An answer discussing "
            "pneumonia findings instead would score low on relevancy."
        ),
    },
    "domain_appropriateness": {
        "name": "Domain Appropriateness",
        "group": "LLM Judge (Domain)",
        "requires_llm": True,
        "formula": "Domain expert LLM judges accuracy & terminology",
        "description": (
            "Does the answer use correct domain terminology? Is the information accurate? "
            "Would a domain expert find the answer acceptable and professional?"
        ),
        "scoring": {
            "1.0": "Excellent terminology and accuracy",
            "0.5": "Some terminology issues or minor inaccuracies",
            "0.0": "Seriously incorrect information",
        },
        "example": (
            "Using correct domain-specific terminology vs. casual/incorrect phrasing — "
            "the former demonstrates appropriate professional language."
        ),
    },
    "citation_accuracy": {
        "name": "Citation Accuracy",
        "group": "LLM Judge (Domain)",
        "requires_llm": True,
        "formula": "LLM verifies that cited sources match the claims they support",
        "description": (
            "Does the answer cite its sources (e.g. [Source 1], report UIDs)? "
            "Do those citations actually match the content of the retrieved documents?"
        ),
        "scoring": {
            "1.0": "All claims properly cited with correct source references",
            "0.5": "Some citations present but incomplete or partially incorrect",
            "0.0": "No citations or completely wrong citations",
        },
        "example": (
            "If the answer says '[Source 2] reports cardiomegaly' but Source 2 actually "
            "discusses pleural effusion, citation accuracy is low."
        ),
    },
    "answer_completeness": {
        "name": "Answer Completeness",
        "group": "LLM Judge (Domain)",
        "requires_llm": True,
        "formula": "LLM compares answer coverage against ground-truth key points",
        "description": (
            "Does the answer cover the key points from the ground-truth reference answer? "
            "Missing important findings or details lowers this score."
        ),
        "scoring": {
            "1.0": "All key points from ground truth are covered",
            "0.5": "Some key points covered, some missing",
            "0.0": "Major key points missing entirely",
        },
        "example": (
            "Ground truth mentions 3 findings (opacity, effusion, cardiomegaly). "
            "If the answer only mentions opacity, completeness ~ 0.33."
        ),
    },
}


def render_metric_guide():
    st.markdown(
        "This page explains each evaluation metric used to assess the RAG pipeline. "
        "Metrics are grouped into three categories."
    )

    # Score color legend
    st.markdown("#### Score color coding")
    legend_cols = st.columns(4)
    for col, (label, color) in zip(legend_cols, [
        ("0.7 - 1.0 (Good)", "green"),
        ("0.4 - 0.69 (Fair)", "orange"),
        ("0.0 - 0.39 (Poor)", "red"),
        ("N/A (failed / skipped)", "gray"),
    ]):
        col.markdown(
            f"<div style='text-align:center;padding:6px;border-radius:4px;"
            f"background:{color};color:white;font-weight:600'>{label}</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Group metrics
    groups = {}
    for key in METRIC_ORDER:
        info = METRIC_GUIDE[key]
        groups.setdefault(info["group"], []).append((key, info))

    for group_name, metrics in groups.items():
        st.subheader(group_name)
        for key, info in metrics:
            llm_badge = " `LLM`" if info["requires_llm"] else " `computed`"
            with st.expander(f"**{info['name']}**{llm_badge}", expanded=True):
                st.markdown(info["description"])

                col_left, col_right = st.columns(2)
                with col_left:
                    st.markdown("**Formula**")
                    st.code(info["formula"], language="text")

                    st.markdown("**Example**")
                    st.info(info["example"])

                with col_right:
                    st.markdown("**Scoring guide**")
                    for score_range, meaning in info["scoring"].items():
                        st.markdown(f"- **{score_range}** — {meaning}")

        st.divider()

    # Pipeline overview
    st.subheader("Pipeline overview")
    st.markdown("""
| Stage | Metrics | Method |
|-------|---------|--------|
| **Retrieval** | Context Precision, Context Recall | Set comparison against golden UIDs (no LLM needed) |
| **Generation** | Faithfulness, Answer Relevancy | LLM evaluates answer vs. context / question |
| **Domain Judge** | Domain Appropriateness, Citation Accuracy, Answer Completeness | LLM-as-expert evaluates against ground truth |

- A score of **-1.0** means the metric was skipped or the LLM call failed. It is displayed as **N/A** and excluded from averages.
- Retrieval metrics can run offline (`--retrieval-only` flag). All other metrics require an LLM backend (Gemini or Ollama).
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.title("Medical Imaging RAG")
    st.caption("Interactive evaluation dashboard & live query interface")

    tab_eval, tab_query, tab_guide = st.tabs(["Eval Dashboard", "RAG Query", "Metric Guide"])

    with tab_eval:
        render_eval_dashboard()

    with tab_query:
        render_rag_query()

    with tab_guide:
        render_metric_guide()


if __name__ == "__main__":
    main()
