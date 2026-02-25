"""Retrieval module: handles text and image queries."""
import base64
import io
from dataclasses import dataclass

import requests
from PIL import Image

import config
from src.embedding import get_client
from src.vector_store import search
from src.prompt_loader import get as get_prompt


@dataclass
class RetrievalResult:
    """Container for retrieval results."""
    query: str
    image_description: str | None
    documents: list[str]
    metadatas: list[dict]
    distances: list[float]

    @property
    def context(self) -> str:
        """Format retrieved documents as context for the generator."""
        context_parts = []
        for i, (doc, meta) in enumerate(zip(self.documents, self.metadatas)):
            source = f"Report {meta.get('uid', 'unknown')} ({meta.get('section', 'unknown')})"
            context_parts.append(f"[Source {i+1}: {source}]\n{doc}")
        return "\n\n".join(context_parts)


def _describe_image_gemini(image: Image.Image) -> str:
    """Use Gemini Vision to generate a medical description of an image."""
    client = get_client()

    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=[get_prompt("image_description_prompt"), image],
    )

    return response.text


def _describe_image_ollama(image: Image.Image) -> str:
    """Use Ollama vision model (e.g. llava) to describe a medical image."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json={
            "model": config.OLLAMA_VISION_MODEL,
            "prompt": get_prompt("image_description_prompt"),
            "images": [b64],
            "stream": False,
            "options": {"temperature": 0.3},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def describe_image(image: Image.Image) -> str:
    """
    Generate a medical description of an image using the configured vision backend.

    Fallback: primary backend fails → try the other → return empty string.
    """
    backend = config.VISION_BACKEND
    primary = _describe_image_ollama if backend == "ollama" else _describe_image_gemini
    fallback = _describe_image_gemini if backend == "ollama" else _describe_image_ollama

    try:
        return primary(image)
    except Exception as e:
        print(f"{backend.title()} vision unavailable ({e}), trying fallback...")
        try:
            return fallback(image)
        except Exception as e2:
            print(f"All vision backends failed ({e2}), skipping image description.")
            return ""


def retrieve(query: str, image: Image.Image | None = None, top_k: int = None) -> RetrievalResult:
    """
    Retrieve relevant medical reports based on text query and optional image.

    For multimodal queries:
    1. If image provided: generate description via configured vision backend
    2. Combine text query + image description
    3. Search vector store for relevant reports

    Args:
        query: Text query from user
        image: Optional uploaded medical image
        top_k: Number of results to return

    Returns:
        RetrievalResult with relevant documents and metadata
    """
    if top_k is None:
        top_k = config.TOP_K

    image_description = None

    # If image is provided, generate a description
    if image is not None:
        backend = config.VISION_BACKEND
        print(f"Generating image description with {backend.title()} Vision...")
        image_description = describe_image(image)
        if image_description:
            print(f"Image description: {image_description[:200]}...")
            # Combine text query with image description for richer retrieval
            combined_query = f"{query}\n\nImage findings: {image_description}"
        else:
            combined_query = query
    else:
        combined_query = query

    # Search vector store
    results = search(combined_query, n_results=top_k)

    return RetrievalResult(
        query=query,
        image_description=image_description,
        documents=results["documents"][0] if results["documents"] else [],
        metadatas=results["metadatas"][0] if results["metadatas"] else [],
        distances=results["distances"][0] if results["distances"] else [],
    )
