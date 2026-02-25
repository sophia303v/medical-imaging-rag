"""
LLM-as-Judge evaluation for domain quality.

Uses a single LLM call per question to evaluate 3 criteria:
- Domain Appropriateness: correct terminology, accurate information
- Citation Accuracy: sources cited and match content
- Answer Completeness: covers key points from ground truth
"""
from dataclasses import dataclass
import json
import re

import requests

import config
from src.prompt_loader import get as get_prompt


@dataclass
class JudgeResult:
    """Result of LLM judge evaluation for a single question."""
    domain_appropriateness: float   # 0.0 to 1.0
    citation_accuracy: float        # 0.0 to 1.0
    answer_completeness: float      # 0.0 to 1.0
    explanation: str

    @property
    def average_score(self) -> float:
        scores = [
            self.domain_appropriateness,
            self.citation_accuracy,
            self.answer_completeness,
        ]
        valid = [s for s in scores if s >= 0]
        return round(sum(valid) / len(valid), 4) if valid else -1.0


def judge_answer(
    question: str,
    answer: str,
    ground_truth: str,
    context: str,
) -> JudgeResult:
    """
    Evaluate an answer using LLM-as-Judge with 3 medical domain criteria.

    Args:
        question: The original question
        answer: The AI-generated answer to evaluate
        ground_truth: The reference/correct answer
        context: The retrieved context used to generate the answer

    Returns:
        JudgeResult with scores for each criterion
    """
    prompt = get_prompt("judge_prompt").format(
        question=question,
        ground_truth=ground_truth,
        answer=answer,
        context=context,
    )

    try:
        if config.GENERATION_BACKEND == "ollama":
            resp = requests.post(
                f"{config.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": config.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 512},
                },
                timeout=120,
            )
            resp.raise_for_status()
            text = resp.json()["response"].strip()
        else:
            from src.embedding import get_client
            client = get_client()
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=[prompt],
                config={"temperature": 0.0, "max_output_tokens": 512},
            )
            text = response.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        parsed = json.loads(text)

        def clamp(v):
            return round(max(0.0, min(1.0, float(v))), 4)

        return JudgeResult(
            domain_appropriateness=clamp(parsed["domain_appropriateness"]),
            citation_accuracy=clamp(parsed["citation_accuracy"]),
            answer_completeness=clamp(parsed["answer_completeness"]),
            explanation=parsed.get("explanation", ""),
        )

    except Exception as e:
        return JudgeResult(
            domain_appropriateness=-1.0,
            citation_accuracy=-1.0,
            answer_completeness=-1.0,
            explanation=f"LLM judge failed: {e}",
        )
