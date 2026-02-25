"""
Evaluation report visualization: generates a self-contained HTML file
with interactive Plotly charts.

Charts included:
1. Radar chart — all 7 metrics at a glance
2. Bar chart — scores grouped by question category
3. Summary table — mean, min, max per metric
4. Detail table — per-question scores with color coding

Also supports comparison reports across multiple experiments.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.evaluation.runner import EvaluationReport


# Metric display names (short labels for charts)
METRIC_LABELS = {
    "context_precision": "Ctx Precision",
    "context_recall": "Ctx Recall",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Relevancy",
    "domain_appropriateness": "Domain",
    "citation_accuracy": "Citation",
    "answer_completeness": "Completeness",
}

METRIC_DESCRIPTIONS = {
    "context_precision": "Retrieved docs 中有多少比例是相關的 (Precision = hits / retrieved)",
    "context_recall": "相關 docs 中有多少被成功檢索到 (Recall = hits / relevant)",
    "reciprocal_rank": "第一個相關文件出現在第幾名 (MRR = 1/rank)",
    "ndcg": "考慮排序的檢索品質，越前面的相關文件分數越高 (Normalized DCG)",
    "faithfulness": "答案是否忠於檢索到的 context，不捏造事實 (LLM 評分)",
    "answer_relevancy": "答案是否切題、有回答到問題 (LLM 評分)",
    "domain_appropriateness": "領域術語是否正確、內容是否合理 (LLM Judge)",
    "citation_accuracy": "是否有引用來源、引用是否正確 (LLM Judge)",
    "answer_completeness": "相比 ground truth，答案的完整度如何 (LLM Judge)",
}

METRIC_ORDER = list(METRIC_LABELS.keys())


def generate_html_report(
    report: EvaluationReport,
    output_path: Path,
    dataset_name: str | None = None,
) -> Path:
    """
    Generate a self-contained HTML report with Plotly charts.

    Args:
        report: EvaluationReport with evaluation results
        output_path: Where to save the HTML file
        dataset_name: Name of the dataset used (shown in report title)

    Returns:
        Path to the generated HTML file
    """
    import config as _cfg
    if dataset_name is None:
        dataset_name = getattr(_cfg, "DATASET_NAME", "unknown")
    agg = report.aggregate_scores()
    by_cat = report.scores_by_category()

    # --- Build chart data ---
    radar_data = _build_radar_data(agg)
    bar_data = _build_bar_data(by_cat)
    summary_rows = _build_summary_table(agg)
    detail_rows = _build_detail_table(report)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG Evaluation Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f5f7fa; color: #1a1a2e; padding: 2rem; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.3rem; }}
  .subtitle {{ color: #666; margin-bottom: 2rem; font-size: 0.95rem; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }}
  .card {{ background: #fff; border-radius: 12px; padding: 1.5rem;
           box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .card h2 {{ font-size: 1.1rem; margin-bottom: 1rem; color: #16213e; }}
  .full-width {{ grid-column: 1 / -1; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ background: #16213e; color: #fff; padding: 0.6rem 0.5rem; text-align: left;
       position: sticky; top: 0; }}
  td {{ padding: 0.5rem; border-bottom: 1px solid #e8e8e8; }}
  tr:hover {{ background: #f0f4ff; }}
  .score-cell {{ text-align: center; font-weight: 600; border-radius: 4px; }}
  .score-high {{ background: #d4edda; color: #155724; }}
  .score-mid {{ background: #fff3cd; color: #856404; }}
  .score-low {{ background: #f8d7da; color: #721c24; }}
  .score-na {{ background: #e2e3e5; color: #6c757d; }}
  .metric-desc {{ color: #888; font-size: 0.78rem; font-weight: 400; }}
  .stats {{ display: flex; gap: 1.5rem; margin-bottom: 1.5rem; flex-wrap: wrap; }}
  .stat-box {{ background: #fff; border-radius: 10px; padding: 1rem 1.5rem;
               box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; min-width: 140px; }}
  .stat-box .value {{ font-size: 1.6rem; font-weight: 700; color: #0f3460; }}
  .stat-box .label {{ font-size: 0.8rem; color: #888; margin-top: 0.2rem; }}
  .detail-scroll {{ max-height: 500px; overflow-y: auto; }}
</style>
</head>
<body>
<div class="container">
  <h1>RAG Evaluation Report — {dataset_name}</h1>
  <p class="subtitle">{len(report.results)} questions evaluated in {report.run_time_seconds:.1f}s | Dataset: {dataset_name} | TOP_K: {getattr(_cfg, 'TOP_K', '?')}</p>

  <!-- Quick stats -->
  <div class="stats">
    {_build_stat_boxes(agg, report)}
  </div>

  <!-- Charts row -->
  <div class="grid">
    <div class="card">
      <h2>Overall Metrics (Radar)</h2>
      <div id="radar-chart"></div>
    </div>
    <div class="card">
      <h2>Scores by Category</h2>
      <div id="bar-chart"></div>
    </div>
  </div>

  <!-- Summary table -->
  <div class="card full-width" style="margin-bottom:1.5rem;">
    <h2>Metric Summary</h2>
    <table>
      <tr><th>Metric</th><th>Mean</th><th>Min</th><th>Max</th><th>Count</th></tr>
      {summary_rows}
    </table>
  </div>

  <!-- Detail table -->
  <div class="card full-width">
    <h2>Per-Question Results</h2>
    <div class="detail-scroll">
    <table>
      <tr>
        <th>ID</th><th>Question</th><th>Category</th><th>Difficulty</th>
        <th>Ctx Prec</th><th>Ctx Rec</th><th>Faith</th><th>Relev</th>
        <th>Domain</th><th>Citation</th><th>Complete</th>
      </tr>
      {detail_rows}
    </table>
    </div>
  </div>
</div>

<script>
// Radar chart
{radar_data}

// Bar chart
{bar_data}
</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"HTML report saved to {output_path}")
    return output_path


def _score_class(score: float) -> str:
    """CSS class for color coding a score."""
    if score < 0:
        return "score-na"
    if score >= 0.7:
        return "score-high"
    if score >= 0.4:
        return "score-mid"
    return "score-low"


def _fmt(score: float) -> str:
    """Format score for display."""
    if score < 0:
        return "N/A"
    return f"{score:.2f}"


def _build_stat_boxes(agg: dict, report: EvaluationReport) -> str:
    """Build quick-stat HTML boxes."""
    # Overall mean across all metrics
    all_means = [v["mean"] for v in agg.values() if v["mean"] >= 0]
    overall = sum(all_means) / len(all_means) if all_means else 0

    retrieval_metrics = ["context_precision", "context_recall"]
    ret_means = [agg[m]["mean"] for m in retrieval_metrics if agg[m]["mean"] >= 0]
    retrieval_avg = sum(ret_means) / len(ret_means) if ret_means else 0

    gen_metrics = ["faithfulness", "answer_relevancy"]
    gen_means = [agg[m]["mean"] for m in gen_metrics if agg[m]["mean"] >= 0]
    gen_avg = sum(gen_means) / len(gen_means) if gen_means else 0

    judge_metrics = ["domain_appropriateness", "citation_accuracy", "answer_completeness"]
    judge_means = [agg[m]["mean"] for m in judge_metrics if agg[m]["mean"] >= 0]
    judge_avg = sum(judge_means) / len(judge_means) if judge_means else 0

    return f"""
    <div class="stat-box"><div class="value">{overall:.2f}</div><div class="label">Overall Score</div></div>
    <div class="stat-box"><div class="value">{retrieval_avg:.2f}</div><div class="label">Retrieval Quality</div></div>
    <div class="stat-box"><div class="value">{gen_avg:.2f}</div><div class="label">Generation Quality</div></div>
    <div class="stat-box"><div class="value">{judge_avg:.2f}</div><div class="label">Domain Judge</div></div>
    <div class="stat-box"><div class="value">{len(report.results)}</div><div class="label">Questions</div></div>
    """


def _build_radar_data(agg: dict) -> str:
    """Build Plotly radar chart JS."""
    labels = [METRIC_LABELS[m] for m in METRIC_ORDER]
    values = [max(0, agg[m]["mean"]) for m in METRIC_ORDER]
    # Close the polygon
    labels_js = json.dumps(labels + [labels[0]])
    values_js = json.dumps(values + [values[0]])

    return f"""
Plotly.newPlot('radar-chart', [{{
  type: 'scatterpolar',
  r: {values_js},
  theta: {labels_js},
  fill: 'toself',
  fillcolor: 'rgba(15,52,96,0.15)',
  line: {{ color: '#0f3460', width: 2 }},
  marker: {{ size: 6, color: '#0f3460' }}
}}], {{
  polar: {{
    radialaxis: {{ visible: true, range: [0, 1], tickvals: [0.2, 0.4, 0.6, 0.8, 1.0] }}
  }},
  margin: {{ t: 30, b: 30, l: 60, r: 60 }},
  height: 350,
  showlegend: false
}}, {{ responsive: true }});
"""


def _build_bar_data(by_cat: dict) -> str:
    """Build Plotly grouped bar chart JS."""
    categories = list(by_cat.keys())
    traces = []
    colors = [
        "#0f3460", "#1a73e8", "#4CAF50", "#FF9800",
        "#9C27B0", "#E91E63", "#00BCD4",
    ]
    for i, metric in enumerate(METRIC_ORDER):
        values = [max(0, by_cat[cat].get(metric, 0)) for cat in categories]
        traces.append({
            "x": categories,
            "y": values,
            "name": METRIC_LABELS[metric],
            "type": "bar",
            "marker": {"color": colors[i % len(colors)]},
        })

    return f"""
Plotly.newPlot('bar-chart', {json.dumps(traces)}, {{
  barmode: 'group',
  yaxis: {{ range: [0, 1], title: 'Score' }},
  xaxis: {{ title: 'Category' }},
  margin: {{ t: 30, b: 60, l: 50, r: 20 }},
  height: 350,
  legend: {{ orientation: 'h', y: -0.25, x: 0.5, xanchor: 'center', font: {{ size: 10 }} }}
}}, {{ responsive: true }});
"""


def _build_summary_table(agg: dict) -> str:
    """Build HTML rows for the summary table."""
    rows = []
    for metric in METRIC_ORDER:
        stats = agg[metric]
        label = METRIC_LABELS[metric]
        desc = METRIC_DESCRIPTIONS.get(metric, "")
        mean_cls = _score_class(stats["mean"])
        rows.append(
            f"<tr><td><strong>{label}</strong><br>"
            f"<span class='metric-desc'>{desc}</span></td>"
            f"<td class='score-cell {mean_cls}'>{_fmt(stats['mean'])}</td>"
            f"<td class='score-cell'>{_fmt(stats['min'])}</td>"
            f"<td class='score-cell'>{_fmt(stats['max'])}</td>"
            f"<td>{stats['count']}</td></tr>"
        )
    return "\n".join(rows)


def _build_detail_table(report: EvaluationReport) -> str:
    """Build HTML rows for the per-question detail table."""
    rows = []
    for r in report.results:
        scores = [
            r.context_precision, r.context_recall, r.faithfulness,
            r.answer_relevancy, r.domain_appropriateness,
            r.citation_accuracy, r.answer_completeness,
        ]
        cells = "".join(
            f"<td class='score-cell {_score_class(s)}'>{_fmt(s)}</td>"
            for s in scores
        )
        # Truncate long questions for display, show full on hover
        q_short = r.question[:60] + "..." if len(r.question) > 60 else r.question
        q_escaped = r.question.replace('"', '&quot;').replace('<', '&lt;')
        rows.append(
            f"<tr><td>{r.question_id}</td>"
            f"<td title=\"{q_escaped}\" style=\"max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;\">{q_short}</td>"
            f"<td>{r.category}</td><td>{r.difficulty}</td>"
            f"{cells}</tr>"
        )
    return "\n".join(rows)


# ──────────────────────────────────────────────────────────────
# Experiment comparison report
# ──────────────────────────────────────────────────────────────

# Colors for up to 8 experiments
_EXP_COLORS = [
    "#0f3460", "#e74c3c", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#3498db",
]


def generate_comparison_report(
    experiments: list[dict],
    output_path: Path,
) -> Path:
    """
    Generate a comparison HTML report across multiple experiments.

    Args:
        experiments: list of dicts, each with keys:
            - "name": display name (str)
            - "results": loaded eval_results.json (dict)
            - "config": loaded config.yaml (dict, optional)
        output_path: where to save the HTML file

    Returns:
        Path to the generated HTML file
    """
    names = [e["name"] for e in experiments]

    # --- Build sections ---
    config_diff_html = _build_config_diff(experiments)
    radar_js = _build_comparison_radar(experiments)
    bar_js = _build_comparison_bar(experiments)
    category_js = _build_category_comparison(experiments)
    summary_html = _build_comparison_summary(experiments)
    delta_html = _build_delta_table(experiments)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Experiment Comparison</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f5f7fa; color: #1a1a2e; padding: 2rem; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.3rem; }}
  .subtitle {{ color: #666; margin-bottom: 2rem; font-size: 0.95rem; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }}
  .card {{ background: #fff; border-radius: 12px; padding: 1.5rem;
           box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .card h2 {{ font-size: 1.1rem; margin-bottom: 1rem; color: #16213e; }}
  .full-width {{ grid-column: 1 / -1; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ background: #16213e; color: #fff; padding: 0.6rem 0.5rem; text-align: left;
       position: sticky; top: 0; }}
  td {{ padding: 0.5rem; border-bottom: 1px solid #e8e8e8; }}
  tr:hover {{ background: #f0f4ff; }}
  .score-cell {{ text-align: center; font-weight: 600; border-radius: 4px; }}
  .score-high {{ background: #d4edda; color: #155724; }}
  .score-mid {{ background: #fff3cd; color: #856404; }}
  .score-low {{ background: #f8d7da; color: #721c24; }}
  .score-na {{ background: #e2e3e5; color: #6c757d; }}
  .delta-pos {{ color: #155724; font-weight: 600; }}
  .delta-neg {{ color: #721c24; font-weight: 600; }}
  .delta-zero {{ color: #6c757d; }}
  .diff-cell {{ background: #fff8e1; font-weight: 600; }}
  .detail-scroll {{ max-height: 500px; overflow-y: auto; }}
  .legend {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }}
  .legend-item {{ display: flex; align-items: center; gap: 0.3rem; font-size: 0.85rem; }}
  .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; }}
</style>
</head>
<body>
<div class="container">
  <h1>Experiment Comparison</h1>
  <p class="subtitle">Comparing {len(experiments)} experiments: {', '.join(names)}</p>

  <!-- Legend -->
  <div class="legend">
    {"".join(f'<div class="legend-item"><div class="legend-dot" style="background:{_EXP_COLORS[i % len(_EXP_COLORS)]}"></div>{name}</div>' for i, name in enumerate(names))}
  </div>

  <!-- Config diff -->
  <div class="card full-width" style="margin-bottom:1.5rem;">
    <h2>Configuration Differences</h2>
    {config_diff_html}
  </div>

  <!-- Metric summary -->
  <div class="card full-width" style="margin-bottom:1.5rem;">
    <h2>Metric Summary</h2>
    {summary_html}
  </div>

  <!-- Charts row -->
  <div class="grid">
    <div class="card">
      <h2>Overall Metrics (Radar)</h2>
      <div id="radar-chart"></div>
    </div>
    <div class="card">
      <h2>Metrics Comparison (Bar)</h2>
      <div id="bar-chart"></div>
    </div>
  </div>

  <!-- Category comparison -->
  <div class="card full-width" style="margin-bottom:1.5rem;">
    <h2>Scores by Category</h2>
    <div id="cat-chart"></div>
  </div>

  <!-- Per-question delta table -->
  <div class="card full-width">
    <h2>Per-Question Score Changes</h2>
    <p style="color:#666; font-size:0.85rem; margin-bottom:0.8rem;">
      Delta = last experiment minus first experiment. Sorted by largest change.
    </p>
    <div class="detail-scroll">
    {delta_html}
    </div>
  </div>
</div>

<script>
{radar_js}
{bar_js}
{category_js}
</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Comparison report saved to {output_path}")
    return output_path


def _build_config_diff(experiments: list[dict]) -> str:
    """Build HTML table showing config differences between experiments."""
    configs = [e.get("config", {}) for e in experiments]
    names = [e["name"] for e in experiments]

    if not any(configs):
        return "<p style='color:#888;'>No config.yaml files found.</p>"

    # Gather all config keys, find differing ones
    skip_keys = {"timestamp", "python_version", "yaml_config_source", "cli_overrides"}
    all_keys = []
    for c in configs:
        for k in c:
            if k not in skip_keys and k not in all_keys:
                all_keys.append(k)

    diff_rows = []
    for key in all_keys:
        values = [str(c.get(key, "—")) for c in configs]
        if len(set(values)) <= 1:
            continue  # same across all experiments, skip
        cells = "".join(
            f"<td class='diff-cell'>{v}</td>" for v in values
        )
        diff_rows.append(f"<tr><td><strong>{key}</strong></td>{cells}</tr>")

    if not diff_rows:
        return "<p style='color:#888;'>All configurations are identical.</p>"

    header = "<th>Parameter</th>" + "".join(f"<th>{n}</th>" for n in names)
    return f"<table><tr>{header}</tr>{''.join(diff_rows)}</table>"


def _build_comparison_radar(experiments: list[dict]) -> str:
    """Build Plotly radar chart JS with one trace per experiment."""
    labels = [METRIC_LABELS[m] for m in METRIC_ORDER]
    labels_js = json.dumps(labels + [labels[0]])

    traces = []
    for i, exp in enumerate(experiments):
        agg = exp["results"]["aggregate_scores"]
        values = [max(0, agg.get(m, {}).get("mean", -1)) for m in METRIC_ORDER]
        values_closed = values + [values[0]]
        color = _EXP_COLORS[i % len(_EXP_COLORS)]
        # Convert hex to rgba for semi-transparent fill
        r_val = int(color[1:3], 16)
        g_val = int(color[3:5], 16)
        b_val = int(color[5:7], 16)
        fill_rgba = f"rgba({r_val},{g_val},{b_val},0.08)"
        traces.append({
            "type": "scatterpolar",
            "r": values_closed,
            "theta": json.loads(labels_js),
            "fill": "toself",
            "fillcolor": fill_rgba,
            "name": exp["name"],
            "line": {"color": color, "width": 2},
            "marker": {"size": 5, "color": color},
        })

    return f"""
Plotly.newPlot('radar-chart', {json.dumps(traces)}, {{
  polar: {{
    radialaxis: {{ visible: true, range: [0, 1], tickvals: [0.2, 0.4, 0.6, 0.8, 1.0] }}
  }},
  margin: {{ t: 30, b: 30, l: 60, r: 60 }},
  height: 380,
  legend: {{ orientation: 'h', y: -0.15, x: 0.5, xanchor: 'center' }}
}}, {{ responsive: true }});
"""


def _build_comparison_bar(experiments: list[dict]) -> str:
    """Build Plotly grouped bar chart: one group per metric, one bar per experiment."""
    labels = [METRIC_LABELS[m] for m in METRIC_ORDER]
    traces = []

    for i, exp in enumerate(experiments):
        agg = exp["results"]["aggregate_scores"]
        values = [max(0, agg.get(m, {}).get("mean", -1)) for m in METRIC_ORDER]
        color = _EXP_COLORS[i % len(_EXP_COLORS)]
        traces.append({
            "x": labels,
            "y": values,
            "name": exp["name"],
            "type": "bar",
            "marker": {"color": color},
        })

    return f"""
Plotly.newPlot('bar-chart', {json.dumps(traces)}, {{
  barmode: 'group',
  yaxis: {{ range: [0, 1], title: 'Score' }},
  margin: {{ t: 30, b: 80, l: 50, r: 20 }},
  height: 380,
  legend: {{ orientation: 'h', y: -0.3, x: 0.5, xanchor: 'center' }}
}}, {{ responsive: true }});
"""


def _build_category_comparison(experiments: list[dict]) -> str:
    """Build Plotly chart comparing category-level scores across experiments."""
    # Collect all categories
    all_cats = []
    for exp in experiments:
        for cat in exp["results"].get("scores_by_category", {}):
            if cat not in all_cats:
                all_cats.append(cat)
    all_cats.sort()

    # Focus on retrieval metrics (most likely to have data)
    focus_metrics = ["context_precision", "context_recall"]
    traces = []

    for i, exp in enumerate(experiments):
        by_cat = exp["results"].get("scores_by_category", {})
        color = _EXP_COLORS[i % len(_EXP_COLORS)]
        for j, metric in enumerate(focus_metrics):
            values = [max(0, by_cat.get(cat, {}).get(metric, 0)) for cat in all_cats]
            dash = "solid" if j == 0 else "dash"
            traces.append({
                "x": all_cats,
                "y": values,
                "name": f"{exp['name']} — {METRIC_LABELS[metric]}",
                "type": "bar",
                "marker": {"color": color, "opacity": 1.0 if j == 0 else 0.5},
            })

    return f"""
Plotly.newPlot('cat-chart', {json.dumps(traces)}, {{
  barmode: 'group',
  yaxis: {{ range: [0, 1], title: 'Score' }},
  xaxis: {{ title: 'Category' }},
  margin: {{ t: 30, b: 60, l: 50, r: 20 }},
  height: 350,
  legend: {{ orientation: 'h', y: -0.25, x: 0.5, xanchor: 'center', font: {{ size: 10 }} }}
}}, {{ responsive: true }});
"""


def _build_comparison_summary(experiments: list[dict]) -> str:
    """Build HTML table comparing aggregate metrics across experiments."""
    names = [e["name"] for e in experiments]
    header = "<th>Metric</th>" + "".join(f"<th>{n}</th>" for n in names)

    # If 2 experiments, add delta column
    show_delta = len(experiments) == 2

    if show_delta:
        header += "<th>Delta</th>"

    rows = []
    for metric in METRIC_ORDER:
        label = METRIC_LABELS[metric]
        cells = ""
        values = []
        for exp in experiments:
            agg = exp["results"]["aggregate_scores"]
            mean = agg.get(metric, {}).get("mean", -1.0)
            values.append(mean)
            cls = _score_class(mean)
            cells += f"<td class='score-cell {cls}'>{_fmt(mean)}</td>"

        if show_delta:
            a, b = values[0], values[1]
            if a >= 0 and b >= 0:
                d = b - a
                cls = "delta-pos" if d > 0.005 else ("delta-neg" if d < -0.005 else "delta-zero")
                sign = "+" if d > 0 else ""
                cells += f"<td class='{cls}'>{sign}{d:.3f}</td>"
            else:
                cells += "<td class='delta-zero'>—</td>"

        rows.append(f"<tr><td>{label}</td>{cells}</tr>")

    return f"<table><tr>{header}</tr>{''.join(rows)}</table>"


def _build_delta_table(experiments: list[dict]) -> str:
    """Build per-question delta table (first vs last experiment)."""
    if len(experiments) < 2:
        return "<p style='color:#888;'>Need at least 2 experiments for delta comparison.</p>"

    first = experiments[0]["results"]
    last = experiments[-1]["results"]

    # Build lookup: question_id -> scores dict
    def _scores_by_id(results_dict: dict) -> dict:
        out = {}
        for r in results_dict.get("results", []):
            out[r["question_id"]] = r
        return out

    first_by_id = _scores_by_id(first)
    last_by_id = _scores_by_id(last)

    all_ids = list(first_by_id.keys())
    focus = ["context_precision", "context_recall"]  # focus on metrics with data

    # Compute deltas
    deltas = []
    for qid in all_ids:
        f_scores = first_by_id.get(qid, {}).get("scores", {})
        l_scores = last_by_id.get(qid, {}).get("scores", {})
        row_deltas = {}
        total_delta = 0
        for m in focus:
            fv = f_scores.get(m, -1)
            lv = l_scores.get(m, -1)
            if fv >= 0 and lv >= 0:
                d = lv - fv
                row_deltas[m] = d
                total_delta += abs(d)
            else:
                row_deltas[m] = None
        info = first_by_id.get(qid, {})
        deltas.append((qid, info.get("category", ""), total_delta, row_deltas, f_scores, l_scores))

    # Sort by total delta descending
    deltas.sort(key=lambda x: -x[2])

    first_name = experiments[0]["name"]
    last_name = experiments[-1]["name"]

    header_cells = "<th>ID</th><th>Category</th>"
    for m in focus:
        short = METRIC_LABELS[m]
        header_cells += f"<th>{first_name}<br>{short}</th><th>{last_name}<br>{short}</th><th>Delta</th>"

    rows = []
    for qid, cat, _, row_deltas, f_scores, l_scores in deltas:
        cells = f"<td>{qid}</td><td>{cat}</td>"
        for m in focus:
            fv = f_scores.get(m, -1)
            lv = l_scores.get(m, -1)
            d = row_deltas[m]
            cells += f"<td class='score-cell {_score_class(fv)}'>{_fmt(fv)}</td>"
            cells += f"<td class='score-cell {_score_class(lv)}'>{_fmt(lv)}</td>"
            if d is not None:
                cls = "delta-pos" if d > 0.005 else ("delta-neg" if d < -0.005 else "delta-zero")
                sign = "+" if d > 0 else ""
                cells += f"<td class='{cls}'>{sign}{d:.3f}</td>"
            else:
                cells += "<td class='delta-zero'>—</td>"
        rows.append(f"<tr>{cells}</tr>")

    return f"<table><tr>{header_cells}</tr>{''.join(rows)}</table>"
