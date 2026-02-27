"""
Causal Chain Extraction module.

Extracts cause-and-effect relationships from text using LLM,
organizes them into causal chains, and formats the output for display.
"""
import json
import requests

import config
from src.prompt_loader import get as get_prompt


def _call_gemini(system_prompt: str, user_prompt: str) -> str:
    """Call Gemini API for causal chain extraction."""
    from src.embedding import get_client
    client = get_client()

    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=[user_prompt],
        config={
            "system_instruction": system_prompt,
            "temperature": 0.2,
            "max_output_tokens": 4096,
        },
    )
    return response.text


def _call_ollama(system_prompt: str, user_prompt: str) -> str:
    """Call Ollama API for causal chain extraction."""
    body = {
        "model": config.OLLAMA_MODEL,
        "prompt": user_prompt,
        "system": system_prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 4096},
    }
    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json=body,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call LLM with fallback chain (same pattern as generator.py)."""
    backend = config.GENERATION_BACKEND
    try:
        if backend == "ollama":
            return _call_ollama(system_prompt, user_prompt)
        else:
            return _call_gemini(system_prompt, user_prompt)
    except Exception as e:
        print(f"{backend.title()} unavailable ({e}), trying fallback...")
        try:
            if backend == "ollama":
                return _call_gemini(system_prompt, user_prompt)
            else:
                return _call_ollama(system_prompt, user_prompt)
        except Exception as e2:
            raise RuntimeError(f"All LLM backends failed: {e2}") from e2


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove markdown code fences
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)
    return json.loads(cleaned)


def extract_causal_chains(article: str) -> dict:
    """
    Extract causal chains from an article.

    Args:
        article: The input article text.

    Returns:
        Dict with keys: causal_pairs, causal_chains, summary
    """
    system_prompt = get_prompt("causal_chain_system_prompt")
    user_prompt = get_prompt("causal_chain_extract_prompt").format(article=article)

    raw_response = _call_llm(system_prompt, user_prompt)

    try:
        result = _parse_json_response(raw_response)
    except (json.JSONDecodeError, ValueError):
        # If JSON parsing fails, return the raw text wrapped in a structure
        result = {
            "causal_pairs": [],
            "causal_chains": [],
            "summary": raw_response,
            "_parse_error": True,
        }

    return result


def format_causal_pairs_table(result: dict) -> str:
    """Format causal pairs as a Markdown table."""
    if result.get("_parse_error"):
        return f"**LLM returned non-JSON response:**\n\n{result['summary']}"

    pairs = result.get("causal_pairs", [])
    if not pairs:
        return "*No causal pairs found.*"

    lines = [
        "| # | Cause | Effect | Confidence | Evidence |",
        "|---|-------|--------|------------|----------|",
    ]
    for p in pairs:
        pid = p.get("id", "")
        cause = p.get("cause", "").replace("|", "\\|")
        effect = p.get("effect", "").replace("|", "\\|")
        conf = p.get("confidence", "")
        evidence = p.get("evidence", "").replace("|", "\\|")
        # Truncate long evidence for table readability
        if len(evidence) > 80:
            evidence = evidence[:77] + "..."
        lines.append(f"| {pid} | {cause} | {effect} | {conf} | {evidence} |")

    return "\n".join(lines)


def format_chains_diagram(result: dict) -> str:
    """Format causal chains as a text-based flow diagram."""
    if result.get("_parse_error"):
        return ""

    chains = result.get("causal_chains", [])
    if not chains:
        return "*No multi-step causal chains identified.*"

    lines = []
    for i, chain_obj in enumerate(chains, 1):
        nodes = chain_obj.get("chain", [])
        desc = chain_obj.get("description", "")
        arrow_chain = " → ".join(nodes)
        lines.append(f"**Chain {i}:** {arrow_chain}")
        if desc:
            lines.append(f"  _{desc}_")
        lines.append("")

    return "\n".join(lines)


def format_mermaid_diagram(result: dict) -> str:
    """Format causal chains as a Mermaid flowchart for visualization."""
    if result.get("_parse_error"):
        return ""

    pairs = result.get("causal_pairs", [])
    if not pairs:
        return ""

    # Collect unique nodes and edges
    node_ids: dict[str, str] = {}
    edges: list[tuple[str, str, str]] = []

    def _node_id(label: str) -> str:
        if label not in node_ids:
            node_ids[label] = f"N{len(node_ids)}"
        return node_ids[label]

    for p in pairs:
        cause = p.get("cause", "")
        effect = p.get("effect", "")
        conf = p.get("confidence", "")
        if cause and effect:
            edges.append((_node_id(cause), _node_id(effect), conf))

    lines = ["graph LR"]
    for label, nid in node_ids.items():
        # Escape quotes in labels
        safe_label = label.replace('"', "'")
        lines.append(f'    {nid}["{safe_label}"]')
    for src, dst, conf in edges:
        if conf == "high":
            lines.append(f"    {src} ==> {dst}")
        elif conf == "low":
            lines.append(f"    {src} -.-> {dst}")
        else:
            lines.append(f"    {src} --> {dst}")

    return "\n".join(lines)


def format_full_output(result: dict) -> tuple[str, str, str]:
    """
    Format the full extraction result.

    Returns:
        Tuple of (pairs_table, chains_text, mermaid_code)
    """
    summary = result.get("summary", "")
    pairs_md = format_causal_pairs_table(result)
    chains_md = format_chains_diagram(result)
    mermaid_code = format_mermaid_diagram(result)

    # Build main output
    main_output = ""
    if summary:
        main_output += f"### Summary\n{summary}\n\n"
    main_output += f"### Causal Pairs\n{pairs_md}\n\n"
    main_output += f"### Causal Chains\n{chains_md}"

    return main_output, mermaid_code, json.dumps(result, ensure_ascii=False, indent=2)
