"""
Generate golden QA pairs from OpenI radiology reports using Gemini.

Usage:
    python scripts/generate_qa.py                          # 50 reports, 2 QA each → ~100 pairs
    python scripts/generate_qa.py --num-reports 20         # Fewer reports
    python scripts/generate_qa.py --questions-per-report 3 # More questions per report
    python scripts/generate_qa.py --dataset openi          # Specify dataset

Selects high-quality reports (long findings, few XXXX placeholders) and generates
factual, comparative, and diagnostic QA pairs compatible with the eval runner.

Output: data/<dataset>/golden_qa.json
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config
from src.prompt_loader import get as get_prompt


def _quality_score(report: dict) -> float:
    """Score a report's quality for QA generation. Higher = better."""
    findings = report.get("findings", "")
    impression = report.get("impression", "")
    text = findings + " " + impression

    # Penalize XXXX placeholders
    xxxx_count = text.lower().count("xxxx")
    # Reward longer findings
    word_count = len(text.split())
    # Penalize empty fields
    if not findings or not impression:
        return 0.0

    return max(0, word_count - xxxx_count * 10)


def select_reports(reports: list[dict], num_reports: int, seed: int = 42) -> list[dict]:
    """Select high-quality reports for QA generation."""
    # Score and sort by quality
    scored = [(r, _quality_score(r)) for r in reports]
    scored = [(r, s) for r, s in scored if s > 20]  # minimum quality threshold
    scored.sort(key=lambda x: -x[1])

    # Take top candidates, then sample for diversity
    top_pool = scored[:min(num_reports * 3, len(scored))]
    random.seed(seed)
    selected = random.sample(top_pool, min(num_reports, len(top_pool)))

    print(f"Selected {len(selected)} reports from {len(scored)} candidates "
          f"(quality range: {selected[-1][1]:.0f}–{selected[0][1]:.0f})")
    return [r for r, _ in selected]




def _call_gemini(client, prompt: str, max_retries: int = 3) -> str:
    """Call Gemini with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config={"temperature": 0.3, "max_output_tokens": 2048},
            )
            return response.text
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Retry {attempt + 1}/{max_retries} after error: {e}")
                time.sleep(wait)
            else:
                raise


def _parse_json_response(text: str) -> list[dict]:
    """Parse JSON from Gemini response, handling markdown fences."""
    # Strip markdown code fences if present
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        return [result]
    except json.JSONDecodeError as e:
        print(f"  Warning: Failed to parse JSON: {e}")
        print(f"  Response snippet: {text[:200]}")
        return []


def generate_factual_qa(client, report: dict, n: int = 2) -> list[dict]:
    """Generate factual QA pairs for a single report."""
    prompt = get_prompt("factual_qa_prompt").format(
        n=n,
        uid=report["uid"],
        indication=report.get("indication", ""),
        findings=report.get("findings", ""),
        impression=report.get("impression", ""),
    )
    text = _call_gemini(client, prompt)
    pairs = _parse_json_response(text)

    # Add report UID reference
    for p in pairs:
        p.setdefault("relevant_report_uids", [report["uid"]])
        p.setdefault("category", "factual")
        p.setdefault("difficulty", "medium")
    return pairs


def generate_comparative_qa(client, reports: list[dict], n: int = 2) -> list[dict]:
    """Generate comparative QA pairs across multiple reports."""
    reports_block = ""
    for r in reports:
        reports_block += f"\nReport UID: {r['uid']}\n"
        reports_block += f"Findings: {r.get('findings', '')}\n"
        reports_block += f"Impression: {r.get('impression', '')}\n"

    prompt = get_prompt("comparative_qa_prompt").format(n=n, reports_block=reports_block)
    text = _call_gemini(client, prompt)
    pairs = _parse_json_response(text)

    for p in pairs:
        p.setdefault("relevant_report_uids", [r["uid"] for r in reports])
        p.setdefault("category", "comparative")
        p.setdefault("difficulty", "hard")
    return pairs


def generate_diagnostic_qa(client, report: dict, n: int = 1) -> list[dict]:
    """Generate diagnostic reasoning QA pairs for a single report."""
    prompt = get_prompt("diagnostic_qa_prompt").format(
        n=n,
        uid=report["uid"],
        indication=report.get("indication", ""),
        findings=report.get("findings", ""),
        impression=report.get("impression", ""),
    )
    text = _call_gemini(client, prompt)
    pairs = _parse_json_response(text)

    for p in pairs:
        p.setdefault("relevant_report_uids", [report["uid"]])
        p.setdefault("category", "diagnostic")
        p.setdefault("difficulty", "medium")
    return pairs


def generate_all_qa(
    dataset_name: str = "openi",
    num_reports: int = 50,
    questions_per_report: int = 2,
    seed: int = 42,
):
    """Main generation pipeline."""
    from src.embedding import get_client

    dataset_dir = config.DATA_DIR / dataset_name
    reports_path = dataset_dir / "reports.json"

    if not reports_path.exists():
        print(f"Error: {reports_path} not found.")
        sys.exit(1)

    with open(reports_path) as f:
        all_reports = json.load(f)
    print(f"Loaded {len(all_reports)} reports from {reports_path}")

    # Select high-quality reports
    selected = select_reports(all_reports, num_reports, seed)

    client = get_client()
    all_qa = []
    qa_id = 0

    # --- Factual QA (50% of reports) ---
    factual_reports = selected[:len(selected) // 2]
    print(f"\nGenerating factual QA from {len(factual_reports)} reports...")
    for i, report in enumerate(factual_reports):
        pairs = generate_factual_qa(client, report, n=questions_per_report)
        for p in pairs:
            p["id"] = f"openi_{qa_id:03d}"
            qa_id += 1
            all_qa.append(p)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(factual_reports)} reports done ({len(all_qa)} QA pairs)")
        time.sleep(0.5)  # Rate limiting

    # --- Diagnostic QA (25% of reports) ---
    diag_reports = selected[len(selected) // 2 : 3 * len(selected) // 4]
    print(f"\nGenerating diagnostic QA from {len(diag_reports)} reports...")
    for i, report in enumerate(diag_reports):
        pairs = generate_diagnostic_qa(client, report, n=questions_per_report)
        for p in pairs:
            p["id"] = f"openi_{qa_id:03d}"
            qa_id += 1
            all_qa.append(p)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(diag_reports)} reports done ({len(all_qa)} QA pairs)")
        time.sleep(0.5)

    # --- Comparative QA (25% of reports, grouped in pairs/triples) ---
    comp_reports = selected[3 * len(selected) // 4:]
    print(f"\nGenerating comparative QA from {len(comp_reports)} reports...")
    for i in range(0, len(comp_reports) - 1, 2):
        group = comp_reports[i:i + 3]  # groups of 2-3
        pairs = generate_comparative_qa(client, group, n=questions_per_report)
        for p in pairs:
            p["id"] = f"openi_{qa_id:03d}"
            qa_id += 1
            all_qa.append(p)
        time.sleep(0.5)

    # Save
    output_path = dataset_dir / "golden_qa.json"
    with open(output_path, "w") as f:
        json.dump(all_qa, f, indent=2)

    # Summary
    categories = {}
    for q in all_qa:
        cat = q.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\nDone! Generated {len(all_qa)} QA pairs → {output_path}")
    print(f"Categories: {categories}")


def main():
    parser = argparse.ArgumentParser(description="Generate golden QA pairs using Gemini")
    parser.add_argument("--dataset", default="openi", help="Dataset name (default: openi)")
    parser.add_argument("--num-reports", type=int, default=50,
                        help="Number of reports to sample (default: 50)")
    parser.add_argument("--questions-per-report", type=int, default=2,
                        help="QA pairs per report (default: 2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    generate_all_qa(
        dataset_name=args.dataset,
        num_reports=args.num_reports,
        questions_per_report=args.questions_per_report,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
