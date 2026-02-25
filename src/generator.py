"""
Generation module: produces answers using retrieved context.
- Gemini backend: full LLM generation with multimodal support
- Ollama backend: local LLM generation via Ollama REST API
- Local backend: returns retrieved context directly (for testing without API)
"""
from dataclasses import dataclass
import requests
from PIL import Image

import config
from src.retriever import RetrievalResult
from src.prompt_loader import get as get_prompt


@dataclass
class GenerationResult:
    """Container for generation results."""
    answer: str
    retrieval: RetrievalResult
    prompt_used: str


def _build_prompt(retrieval: RetrievalResult) -> str:
    """Build the user prompt from retrieval results."""
    prompt = get_prompt("user_prompt").format(
        context=retrieval.context,
        query=retrieval.query,
    )

    if retrieval.image_description:
        prompt += get_prompt("user_prompt_image_suffix").format(
            image_description=retrieval.image_description,
        )

    return prompt


def _generate_gemini(prompt: str, image: Image.Image | None = None) -> str:
    """Generate answer using Gemini API."""
    from src.embedding import get_client
    client = get_client()

    contents = [prompt]
    if image is not None:
        contents.append(image)

    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=contents,
        config={
            "system_instruction": get_prompt("system_prompt"),
            "temperature": 0.3,
            "max_output_tokens": 1024,
        },
    )
    return response.text


def _generate_ollama(prompt: str, image: Image.Image | None = None) -> str:
    """Generate answer using a local Ollama model.

    If an image is provided and VISION_BACKEND is "ollama", the image is sent
    as base64 so that a multimodal model (e.g. llava) can incorporate it.
    """
    import base64, io

    body: dict = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "system": get_prompt("system_prompt"),
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 1024},
    }

    if image is not None and config.VISION_BACKEND == "ollama":
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        body["images"] = [base64.b64encode(buf.getvalue()).decode()]

    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json=body,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def _generate_local(prompt: str, retrieval: RetrievalResult) -> str:
    """
    Local fallback: format retrieved context as an answer.
    Used when Gemini API is not available (offline testing).
    """
    answer = f"**Query:** {retrieval.query}\n\n"
    answer += "**Retrieved Evidence:**\n\n"

    for i, (doc, meta, dist) in enumerate(zip(
        retrieval.documents, retrieval.metadatas, retrieval.distances
    )):
        relevance = f"{(1 - dist) * 100:.1f}%"
        uid = meta.get("uid", "unknown")
        section = meta.get("section", "unknown")
        answer += f"**[Source {i+1}]** Report {uid} ({section}) — Relevance: {relevance}\n"
        answer += f"> {doc}\n\n"

    answer += "\n*[Local mode — connect Gemini API for AI-generated analysis]*"
    return answer


def generate_answer(
    retrieval: RetrievalResult,
    image: Image.Image | None = None,
) -> GenerationResult:
    """Generate an answer using retrieved context."""
    prompt = _build_prompt(retrieval)

    backend = config.GENERATION_BACKEND
    try:
        if backend == "ollama":
            answer = _generate_ollama(prompt, image)
        else:
            answer = _generate_gemini(prompt, image)
    except Exception as e:
        # Fallback chain: try the other LLM backend, then local
        print(f"{backend.title()} unavailable ({e}), trying fallback...")
        try:
            if backend == "ollama":
                answer = _generate_gemini(prompt, image)
            else:
                answer = _generate_ollama(prompt, image)
        except Exception as e2:
            print(f"All LLM backends failed ({e2}), using local fallback.")
            answer = _generate_local(prompt, retrieval)

    disclaimer = get_prompt("disclaimer")
    if "disclaimer" not in answer.lower():
        answer += disclaimer

    return GenerationResult(
        answer=answer,
        retrieval=retrieval,
        prompt_used=prompt,
    )
