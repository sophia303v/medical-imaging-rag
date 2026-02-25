"""
Download and convert benchmark datasets into the project's standard format.

Usage:
    python scripts/download_datasets.py squad_v2              # SQuAD 2.0 dev set
    python scripts/download_datasets.py scifact               # BEIR/SciFact
    python scripts/download_datasets.py radqa                 # RadQA (needs PhysioNet credentials)
    python scripts/download_datasets.py all                   # All of the above
    python scripts/download_datasets.py squad_v2 --max-samples 500  # Limit QA pairs

Output format (per dataset):
    data/<dataset>/passages.json   — list of {"uid", "indication", "findings", "impression", "image_paths"}
    data/<dataset>/golden_qa.json  — list of {"id", "question", "ground_truth_answer",
                                              "relevant_report_uids", "category", "difficulty"}
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# Add project root to path so we can import config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_passage(uid: str, text: str) -> dict:
    """Create a passage dict compatible with MedicalReport / data_loader."""
    return {
        "uid": uid,
        "indication": "",
        "findings": text,
        "impression": "",
        "image_paths": [],
    }


def _make_qa(qid: str, question: str, answer: str, uids: list[str],
             category: str = "unknown", difficulty: str = "unknown") -> dict:
    return {
        "id": qid,
        "question": question,
        "ground_truth_answer": answer,
        "relevant_report_uids": uids,
        "category": category,
        "difficulty": difficulty,
    }


def _save_dataset(name: str, passages: list[dict], qa_pairs: list[dict]):
    """Save passages.json and golden_qa.json to data/<name>/."""
    out_dir = config.DATA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    passages_path = out_dir / "passages.json"
    qa_path = out_dir / "golden_qa.json"

    with open(passages_path, "w") as f:
        json.dump(passages, f, indent=2)
    print(f"  Saved {len(passages)} passages → {passages_path}")

    with open(qa_path, "w") as f:
        json.dump(qa_pairs, f, indent=2)
    print(f"  Saved {len(qa_pairs)} QA pairs → {qa_path}")


def _stable_uid(prefix: str, text: str) -> str:
    """Generate a short deterministic UID from text content."""
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{prefix}_{h}"


# ---------------------------------------------------------------------------
# SQuAD 2.0
# ---------------------------------------------------------------------------

def download_squad_v2(max_samples: int | None = 2000):
    """Download SQuAD 2.0 validation split and convert to project format."""
    from datasets import load_dataset

    print("Downloading SQuAD 2.0 (validation split)...")
    ds = load_dataset("rajpurkar/squad_v2", split="validation")

    # Deduplicate contexts → passages
    context_to_uid: dict[str, str] = {}
    passages = []

    for row in ds:
        ctx = row["context"]
        if ctx not in context_to_uid:
            uid = _stable_uid("squad", ctx)
            context_to_uid[ctx] = uid
            passages.append(_make_passage(uid, ctx))

    # Convert QA pairs
    qa_pairs = []
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break

        uid = context_to_uid[row["context"]]
        answers = row["answers"]["text"]
        is_impossible = len(answers) == 0

        qa_pairs.append(_make_qa(
            qid=f"sq_{i:05d}",
            question=row["question"],
            answer=answers[0] if answers else "",
            uids=[uid],
            category="unanswerable" if is_impossible else "squad_v2",
            difficulty="unknown",
        ))

    print(f"  {len(passages)} unique contexts, {len(qa_pairs)} QA pairs "
          f"(max_samples={max_samples})")
    _save_dataset("squad_v2", passages, qa_pairs)


# ---------------------------------------------------------------------------
# SciFact (BEIR)
# ---------------------------------------------------------------------------

def download_scifact(max_samples: int | None = None):
    """Download SciFact via BEIR zip (corpus, queries, qrels) and convert."""
    import io
    import csv
    import zipfile
    import urllib.request

    beir_url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"

    print("Downloading SciFact from BEIR...")
    with urllib.request.urlopen(beir_url) as resp:
        zip_bytes = resp.read()

    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))

    # Parse corpus (JSONL)
    print("  Parsing corpus...")
    corpus_id_to_uid: dict[str, str] = {}
    passages = []
    with zf.open("scifact/corpus.jsonl") as f:
        for line in f:
            row = json.loads(line)
            cid = str(row["_id"])
            title = row.get("title", "")
            text = row.get("text", "")
            full_text = f"{title}\n{text}".strip() if title else text
            uid = f"scifact_{cid}"
            corpus_id_to_uid[cid] = uid
            passages.append(_make_passage(uid, full_text))

    # Parse queries (JSONL)
    print("  Parsing queries...")
    query_map: dict[str, str] = {}
    with zf.open("scifact/queries.jsonl") as f:
        for line in f:
            row = json.loads(line)
            query_map[str(row["_id"])] = row["text"]

    # Parse qrels (TSV) — test split
    print("  Parsing qrels...")
    qrels: dict[str, list[tuple[str, int]]] = {}
    with zf.open("scifact/qrels/test.tsv") as f:
        reader = csv.DictReader(io.TextIOWrapper(f), delimiter="\t")
        for row in reader:
            qid = str(row["query-id"])
            cid = str(row["corpus-id"])
            score = int(row.get("score", 1))
            qrels.setdefault(qid, []).append((cid, score))

    # QA pairs from qrels
    qa_pairs = []
    for i, (qid, rels) in enumerate(sorted(qrels.items())):
        if max_samples is not None and i >= max_samples:
            break

        if qid not in query_map:
            continue

        question = query_map[qid]
        relevant_uids = [
            corpus_id_to_uid[cid]
            for cid, score in sorted(rels, key=lambda x: -x[1])
            if cid in corpus_id_to_uid
        ]

        qa_pairs.append(_make_qa(
            qid=f"sf_{i:04d}",
            question=question,
            answer="",  # SciFact is retrieval-only — no extractive answer
            uids=relevant_uids,
            category="scifact",
            difficulty="unknown",
        ))

    print(f"  {len(passages)} corpus docs, {len(qa_pairs)} test queries")
    _save_dataset("scifact", passages, qa_pairs)


# ---------------------------------------------------------------------------
# RadQA (PhysioNet)
# ---------------------------------------------------------------------------

def download_radqa(max_samples: int | None = None):
    """Download RadQA from PhysioNet (requires credentialed access).

    Expects PHYSIONET_USERNAME and PHYSIONET_PASSWORD env vars,
    or --username / --password CLI args.
    """
    username = os.getenv("PHYSIONET_USERNAME", "")
    password = os.getenv("PHYSIONET_PASSWORD", "")

    if not username or not password:
        print("RadQA requires PhysioNet credentialed access.")
        print("Set PHYSIONET_USERNAME and PHYSIONET_PASSWORD environment variables,")
        print("or use: python scripts/download_datasets.py radqa --username USER --password PASS")
        print("\nTo get access: https://physionet.org/content/radqa/1.0.0/")
        sys.exit(1)

    raw_dir = config.RAW_DATA_DIR / "radqa"
    raw_dir.mkdir(parents=True, exist_ok=True)

    url = "https://physionet.org/files/radqa/1.0.0/"
    print(f"Downloading RadQA from PhysioNet to {raw_dir}...")

    result = subprocess.run(
        ["wget", "-r", "-N", "-c", "-np", "--user", username, "--password", password,
         "-P", str(raw_dir), url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"wget failed (exit {result.returncode}):")
        print(result.stderr[:500])
        sys.exit(1)

    # Find the downloaded JSON files (SQuAD format)
    radqa_files = list(raw_dir.rglob("*.json"))
    if not radqa_files:
        print(f"No JSON files found in {raw_dir}. Check download.")
        sys.exit(1)

    # Parse SQuAD-format JSONs and merge
    all_passages: dict[str, dict] = {}  # context_hash → passage
    all_qa: list[dict] = []
    qa_count = 0

    for json_file in sorted(radqa_files):
        print(f"  Parsing {json_file.name}...")
        with open(json_file) as f:
            data = json.load(f)

        for article in data.get("data", []):
            doc_id = article.get("title", article.get("id", json_file.stem))
            for para in article.get("paragraphs", []):
                ctx = para["context"]
                uid = _stable_uid("radqa", ctx)

                if uid not in all_passages:
                    all_passages[uid] = _make_passage(uid, ctx)

                for qa in para.get("qas", []):
                    if max_samples is not None and qa_count >= max_samples:
                        break

                    answers = qa.get("answers", [])
                    is_impossible = qa.get("is_impossible", len(answers) == 0)
                    answer_text = answers[0]["text"] if answers else ""

                    all_qa.append(_make_qa(
                        qid=f"rq_{qa_count:05d}",
                        question=qa["question"],
                        answer=answer_text,
                        uids=[uid],
                        category="unanswerable" if is_impossible else "radqa",
                        difficulty="unknown",
                    ))
                    qa_count += 1

                if max_samples is not None and qa_count >= max_samples:
                    break
            if max_samples is not None and qa_count >= max_samples:
                break
        if max_samples is not None and qa_count >= max_samples:
            break

    passages = list(all_passages.values())
    print(f"  {len(passages)} unique contexts, {len(all_qa)} QA pairs")
    _save_dataset("radqa", passages, all_qa)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DATASETS = {
    "squad_v2": download_squad_v2,
    "scifact": download_scifact,
    "radqa": download_radqa,
}


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert benchmark datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "dataset", choices=[*DATASETS.keys(), "all"],
        help="Which dataset to download (or 'all')",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum number of QA pairs to keep (default: 2000 for SQuAD, unlimited for others)",
    )
    parser.add_argument("--username", type=str, default=None, help="PhysioNet username (RadQA)")
    parser.add_argument("--password", type=str, default=None, help="PhysioNet password (RadQA)")
    args = parser.parse_args()

    # Set PhysioNet credentials from CLI if provided
    if args.username:
        os.environ["PHYSIONET_USERNAME"] = args.username
    if args.password:
        os.environ["PHYSIONET_PASSWORD"] = args.password

    if args.dataset == "all":
        targets = list(DATASETS.keys())
    else:
        targets = [args.dataset]

    for name in targets:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        max_s = args.max_samples
        if max_s is None and name == "squad_v2":
            max_s = 2000  # sensible default for SQuAD
        DATASETS[name](max_samples=max_s)

    print(f"\nDone! To ingest a dataset, run:")
    print(f"  1. Delete data/chroma_db/ if switching embeddings")
    print(f"  2. DATASET_NAME=<name> python ingest.py")
    print(f"  3. python run_eval.py --dataset <name> --retrieval-only")


if __name__ == "__main__":
    main()
