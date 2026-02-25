"""
CLI entry point for running RAG evaluation.

Usage:
    python run_eval.py                   # Full evaluation (needs Gemini API key)
    python run_eval.py --retrieval-only  # Retrieval metrics only (no API key needed)
    python run_eval.py --qa data/golden_qa.json  # Custom QA file

    # Experiment management:
    python run_eval.py --config experiments/configs/example_topk5.yaml --retrieval-only
    python run_eval.py --top-k 5 --experiment-name topk5_test --retrieval-only
    python run_eval.py --config experiments/configs/topk5.yaml --top-k 7  # CLI overrides YAML

    # Benchmark datasets:
    python run_eval.py --dataset squad_v2 --retrieval-only
    python run_eval.py --dataset scifact --retrieval-only --experiment-name scifact_baseline
"""
import argparse
import json
from datetime import date
from pathlib import Path

import config
from src.evaluation.runner import run_evaluation
from src.evaluation.visualization import generate_html_report


def _resolve_experiment_dir(name: str | None) -> Path | None:
    """Create a unique experiment directory under experiments/.

    Returns None if no experiment name is given (backward-compatible mode).
    Handles collisions by appending _run2, _run3, etc.
    """
    if name is None:
        return None

    base_name = f"{date.today().isoformat()}_{name}"
    exp_dir = config.EXPERIMENTS_DIR / base_name

    if not exp_dir.exists():
        exp_dir.mkdir(parents=True)
        return exp_dir

    # Collision: append _run2, _run3, ...
    run = 2
    while True:
        candidate = config.EXPERIMENTS_DIR / f"{base_name}_run{run}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        run += 1


def _save_config_snapshot(exp_dir: Path, yaml_path: str | None, cli_overrides: dict):
    """Save a YAML snapshot of all experiment parameters to exp_dir/config.yaml."""
    import yaml

    snapshot = config.get_param_snapshot()
    snapshot["yaml_config_source"] = str(yaml_path) if yaml_path else None
    snapshot["cli_overrides"] = cli_overrides

    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--qa", type=str, default=None,
        help="Path to golden QA JSON file (default: data/golden_qa.json)",
    )
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="Only run retrieval metrics (no Gemini API key needed)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results (default: data/)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    # Experiment management flags
    parser.add_argument(
        "--config", type=str, default=None, dest="yaml_config",
        help="Path to YAML config file to override parameters",
    )
    parser.add_argument(
        "--experiment-name", type=str, default=None,
        help="Experiment name (creates experiments/{date}_{name}/ folder)",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Override TOP_K retrieval count",
    )
    parser.add_argument(
        "--embedding-backend", type=str, default=None,
        choices=["sentence-transformers", "tfidf", "gemini"],
        help="Override EMBEDDING_BACKEND",
    )
    parser.add_argument(
        "--generation-backend", type=str, default=None,
        choices=["gemini", "ollama"],
        help="Override GENERATION_BACKEND",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Dataset name to evaluate (e.g. squad_v2, scifact, radqa). "
             "Note: switching datasets requires rebuilding chroma_db first.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit evaluation to first N questions (useful for quick tests)",
    )
    args = parser.parse_args()

    # --- Apply overrides in priority order: YAML < CLI ---
    if args.yaml_config:
        applied = config.load_yaml_config(args.yaml_config)
        if not args.quiet:
            print(f"Loaded YAML config: {args.yaml_config}")
            for k, v in applied.items():
                print(f"  {k} = {v}")

    # CLI flags override YAML
    cli_overrides = {}
    if args.top_k is not None:
        config.TOP_K = args.top_k
        cli_overrides["TOP_K"] = args.top_k
    if args.embedding_backend is not None:
        config.EMBEDDING_BACKEND = args.embedding_backend
        cli_overrides["EMBEDDING_BACKEND"] = args.embedding_backend
    if args.generation_backend is not None:
        config.GENERATION_BACKEND = args.generation_backend
        cli_overrides["GENERATION_BACKEND"] = args.generation_backend

    if args.dataset is not None:
        config.set_dataset(args.dataset)
        cli_overrides["DATASET_NAME"] = args.dataset
        if not args.quiet:
            print(f"Dataset: {args.dataset} ({config.DATASET_DIR})")
            if not config.DATASET_DIR.exists():
                print(f"  Warning: {config.DATASET_DIR} does not exist yet. "
                      f"Run: python scripts/download_datasets.py {args.dataset}")

    if cli_overrides and not args.quiet:
        print("CLI overrides:")
        for k, v in cli_overrides.items():
            print(f"  {k} = {v}")

    # --- Determine output directory ---
    if args.output_dir:
        # Explicit --output-dir takes highest priority
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    elif args.yaml_config or args.experiment_name:
        # Experiment mode: auto-create experiment folder
        exp_name = args.experiment_name or Path(args.yaml_config).stem
        output_dir = _resolve_experiment_dir(exp_name)
        if not args.quiet:
            print(f"\nExperiment directory: {output_dir}")
    else:
        # Backward-compatible: save to data/
        output_dir = config.DATA_DIR

    # Save config snapshot if in experiment mode
    if args.yaml_config or args.experiment_name:
        _save_config_snapshot(output_dir, args.yaml_config, cli_overrides)

    qa_path = Path(args.qa) if args.qa else None
    use_llm = not args.retrieval_only

    if use_llm and not config.GEMINI_API_KEY:
        print("Warning: No GEMINI_API_KEY set. Falling back to retrieval-only mode.")
        print("Set GEMINI_API_KEY in .env for full evaluation (faithfulness, relevancy, LLM judge).\n")
        use_llm = False

    # Run evaluation
    report = run_evaluation(
        golden_qa_path=qa_path,
        use_llm_metrics=use_llm,
        verbose=not args.quiet,
        max_samples=args.max_samples,
    )

    # Save results
    json_path = output_dir / "eval_results.json"
    report.save_json(json_path)

    # Generate HTML report
    html_path = output_dir / "eval_report.html"
    generate_html_report(report, html_path)

    print(f"\nDone! Open {html_path} in a browser to view the report.")


if __name__ == "__main__":
    main()
