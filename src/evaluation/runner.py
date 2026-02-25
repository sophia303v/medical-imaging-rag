"""
Evaluation runner: orchestrates RAGAS metrics and LLM judge across all golden QA pairs.

Produces an EvaluationReport with per-question and aggregate scores.
"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

import config
from src.rag_pipeline import MedicalImagingRAG
from src.embedding import embed_texts
from src.vector_store import get_chroma_client, get_or_create_collection
from src.retriever import RetrievalResult
from src.generator import generate_answer
from src.evaluation.metrics import (
    MetricResult,
    context_precision,
    context_recall,
    reciprocal_rank,
    ndcg_at_k,
    faithfulness,
    answer_relevancy,
)
from src.evaluation.llm_judge import JudgeResult, judge_answer


@dataclass
class QuestionResult:
    """All evaluation results for a single question."""
    question_id: str
    question: str
    category: str
    difficulty: str
    answer: str
    ground_truth: str
    retrieved_uids: list[str]
    relevant_uids: list[str]
    # RAGAS metrics
    context_precision: float
    context_recall: float
    reciprocal_rank: float
    ndcg: float
    faithfulness: float
    answer_relevancy: float
    # LLM judge metrics
    domain_appropriateness: float
    citation_accuracy: float
    answer_completeness: float

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "category": self.category,
            "difficulty": self.difficulty,
            "answer": self.answer,
            "ground_truth": self.ground_truth,
            "retrieved_uids": self.retrieved_uids,
            "relevant_uids": self.relevant_uids,
            "scores": {
                "context_precision": self.context_precision,
                "context_recall": self.context_recall,
                "reciprocal_rank": self.reciprocal_rank,
                "ndcg": self.ndcg,
                "faithfulness": self.faithfulness,
                "answer_relevancy": self.answer_relevancy,
                "domain_appropriateness": self.domain_appropriateness,
                "citation_accuracy": self.citation_accuracy,
                "answer_completeness": self.answer_completeness,
            },
        }


@dataclass
class EvaluationReport:
    """Complete evaluation report across all questions."""
    results: list[QuestionResult] = field(default_factory=list)
    run_time_seconds: float = 0.0

    def aggregate_scores(self) -> dict[str, dict[str, float]]:
        """Compute mean, min, max for each metric across all questions."""
        metric_names = [
            "context_precision", "context_recall", "reciprocal_rank", "ndcg",
            "faithfulness", "answer_relevancy", "domain_appropriateness",
            "citation_accuracy", "answer_completeness",
        ]
        agg = {}
        for name in metric_names:
            values = [
                getattr(r, name) for r in self.results
                if getattr(r, name) >= 0  # skip failed metrics (-1)
            ]
            if values:
                agg[name] = {
                    "mean": round(sum(values) / len(values), 4),
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                    "count": len(values),
                }
            else:
                agg[name] = {"mean": -1.0, "min": -1.0, "max": -1.0, "count": 0}
        return agg

    def scores_by_category(self) -> dict[str, dict[str, float]]:
        """Compute mean scores grouped by question category."""
        categories: dict[str, list[QuestionResult]] = {}
        for r in self.results:
            categories.setdefault(r.category, []).append(r)

        metric_names = [
            "context_precision", "context_recall", "reciprocal_rank", "ndcg",
            "faithfulness", "answer_relevancy", "domain_appropriateness",
            "citation_accuracy", "answer_completeness",
        ]
        result = {}
        for cat, items in sorted(categories.items()):
            cat_scores = {}
            for name in metric_names:
                values = [getattr(r, name) for r in items if getattr(r, name) >= 0]
                cat_scores[name] = round(sum(values) / len(values), 4) if values else -1.0
            result[cat] = cat_scores
        return result

    def to_dict(self) -> dict:
        return {
            "run_time_seconds": round(self.run_time_seconds, 2),
            "num_questions": len(self.results),
            "aggregate_scores": self.aggregate_scores(),
            "scores_by_category": self.scores_by_category(),
            "results": [r.to_dict() for r in self.results],
        }

    def save_json(self, path: Path):
        """Save report to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Results saved to {path}")


def _extract_uids(metadatas: list[dict]) -> list[str]:
    """Extract unique report UIDs from retrieval metadata."""
    uids = []
    seen = set()
    for meta in metadatas:
        uid = meta.get("uid", "")
        if uid and uid not in seen:
            uids.append(uid)
            seen.add(uid)
    return uids


def run_evaluation(
    golden_qa_path: Path | None = None,
    use_llm_metrics: bool = True,
    verbose: bool = True,
    max_samples: int | None = None,
) -> EvaluationReport:
    """
    Run full evaluation on the golden QA dataset.

    Args:
        golden_qa_path: Path to golden QA JSON file
        use_llm_metrics: Whether to run LLM-based metrics (faithfulness,
                         answer_relevancy, LLM judge). Set False to only
                         run retrieval metrics without an API key.
        verbose: Print progress
        max_samples: Limit to first N questions (None = all)

    Returns:
        EvaluationReport with all results
    """
    if golden_qa_path is None:
        golden_qa_path = config.GOLDEN_QA_PATH

    with open(golden_qa_path) as f:
        qa_pairs = json.load(f)

    if max_samples is not None:
        qa_pairs = qa_pairs[:max_samples]
        if verbose:
            print(f"Limited to first {max_samples} questions.")

    if verbose:
        print(f"Loaded {len(qa_pairs)} QA pairs from {golden_qa_path}")

    # Initialize RAG pipeline
    rag = MedicalImagingRAG()
    if not rag._is_ingested:
        if verbose:
            print("No index found. Running ingestion...")
        rag.ingest()

    report = EvaluationReport()
    start_time = time.time()

    # --- Batch embed all queries upfront for speed ---
    all_questions = [qa["question"] for qa in qa_pairs]
    if verbose:
        print("Batch embedding all queries...")
    all_embeddings = embed_texts(all_questions, task_type="RETRIEVAL_QUERY")
    if verbose:
        print(f"Embedded {len(all_embeddings)} queries.")

    # Get ChromaDB collection once
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    pbar = tqdm(qa_pairs, desc="Evaluating", disable=not verbose)
    for i, qa in enumerate(pbar):
        qid = qa["id"]
        question = qa["question"]
        ground_truth = qa["ground_truth_answer"]
        relevant_uids = qa.get("relevant_report_uids", [])
        category = qa.get("category", "unknown")
        difficulty = qa.get("difficulty", "unknown")

        pbar.set_postfix_str(f"{qid}: {question[:40]}")

        # Retrieve using pre-computed embedding
        try:
            results = collection.query(
                query_embeddings=[all_embeddings[i]],
                n_results=config.TOP_K,
                include=["documents", "metadatas", "distances"],
            )
            documents = results["documents"][0] if results["documents"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            retrieved_uids = _extract_uids(metadatas)

            # Build context string
            context_parts = []
            for j, (doc, meta) in enumerate(zip(documents, metadatas)):
                source = f"Report {meta.get('uid', 'unknown')} ({meta.get('section', 'unknown')})"
                context_parts.append(f"[Source {j+1}: {source}]\n{doc}")
            context_text = "\n\n".join(context_parts)
            answer = ""  # retrieval-only doesn't need generation
        except Exception as e:
            if verbose:
                tqdm.write(f"  Query failed: {e}")
            answer = f"[Query failed: {e}]"
            retrieved_uids = []
            context_text = ""

        # --- Retrieval metrics (always run) ---
        cp = context_precision(retrieved_uids, relevant_uids)
        cr = context_recall(retrieved_uids, relevant_uids)
        rr = reciprocal_rank(retrieved_uids, relevant_uids)
        ndcg = ndcg_at_k(retrieved_uids, relevant_uids)

        # --- LLM metrics (optional) ---
        faith_score = -1.0
        relevancy_score = -1.0
        med_approp = -1.0
        cite_acc = -1.0
        completeness = -1.0

        if use_llm_metrics:
            # Generate answer using already-retrieved context (no re-embedding)
            try:
                retrieval = RetrievalResult(
                    query=question,
                    image_description=None,
                    documents=documents,
                    metadatas=metadatas,
                    distances=[],
                )
                gen_result = generate_answer(retrieval)
                answer = gen_result.answer
            except Exception as e:
                answer = f"[Generation failed: {e}]"

            # Run faithfulness, relevancy, and judge concurrently
            with ThreadPoolExecutor(max_workers=3) as pool:
                f_faith = pool.submit(faithfulness, question, answer, context_text)
                f_rel = pool.submit(answer_relevancy, question, answer)
                f_judge = pool.submit(judge_answer, question, answer, ground_truth, context_text)

                faith = f_faith.result()
                relevancy = f_rel.result()
                judge = f_judge.result()

            faith_score = faith.score
            relevancy_score = relevancy.score
            med_approp = judge.domain_appropriateness
            cite_acc = judge.citation_accuracy
            completeness = judge.answer_completeness

        result = QuestionResult(
            question_id=qid,
            question=question,
            category=category,
            difficulty=difficulty,
            answer=answer,
            ground_truth=ground_truth,
            retrieved_uids=retrieved_uids,
            relevant_uids=relevant_uids,
            context_precision=cp.score,
            context_recall=cr.score,
            reciprocal_rank=rr.score,
            ndcg=ndcg.score,
            faithfulness=faith_score,
            answer_relevancy=relevancy_score,
            domain_appropriateness=med_approp,
            citation_accuracy=cite_acc,
            answer_completeness=completeness,
        )
        report.results.append(result)

    report.run_time_seconds = time.time() - start_time

    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluation complete in {report.run_time_seconds:.1f}s")
        print(f"{'='*60}")
        agg = report.aggregate_scores()
        for name, stats in agg.items():
            if stats["count"] > 0:
                print(f"  {name:30s}  mean={stats['mean']:.3f}  "
                      f"min={stats['min']:.3f}  max={stats['max']:.3f}")
            else:
                print(f"  {name:30s}  (no data)")

    return report
