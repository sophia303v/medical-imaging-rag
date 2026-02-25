"""Centralized configuration for Medical Imaging RAG."""
import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DATASET_NAME = os.getenv("DATASET_NAME", "openi_synthetic")
DATASET_DIR = DATA_DIR / DATASET_NAME
CHROMA_DIR = DATA_DIR / "chroma_db"
MODELS_DIR = Path("/Users/sophia/Desktop/CV/models")
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "text-embedding-004"

# Embedding backend: "sentence-transformers" (default) | "tfidf" (offline) | "gemini" (API)
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers")
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L12-v2"

# Generation backend: "gemini" (default) or "ollama" (local)
GENERATION_BACKEND = os.getenv("GENERATION_BACKEND", "gemini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Vision backend: "gemini" (default) or "ollama" (local, uses llava)
VISION_BACKEND = os.getenv("VISION_BACKEND", "gemini")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava:7b")

# Prompt template: name of a YAML file under prompts/ (without .yaml extension)
# Available: "default" (domain-agnostic), "medical" (radiology-specific)
PROMPT_TEMPLATE = os.getenv("PROMPT_TEMPLATE", "default")

# RAG Settings
CHUNK_SIZE = 512          # tokens per chunk (approx)
CHUNK_OVERLAP = 50        # overlap between chunks
TOP_K = 3                 # number of retrieved documents
COLLECTION_NAME = "medical_reports"

# Embedding
EMBEDDING_DIMENSION = 384  # all-MiniLM-L12-v2 output dimension

# Evaluation
GOLDEN_QA_PATH = DATASET_DIR / "golden_qa.json"
EVAL_RESULTS_PATH = DATA_DIR / "eval_results.json"
EVAL_REPORT_PATH = DATA_DIR / "eval_report.html"

# --- Experiment parameter management ---

# Parameters that can be overridden by YAML config (excludes paths and API keys)
EXPERIMENT_PARAMS = [
    "GEMINI_MODEL",
    "EMBEDDING_MODEL",
    "EMBEDDING_BACKEND",
    "SENTENCE_TRANSFORMER_MODEL",
    "GENERATION_BACKEND",
    "OLLAMA_MODEL",
    "OLLAMA_BASE_URL",
    "VISION_BACKEND",
    "OLLAMA_VISION_MODEL",
    "PROMPT_TEMPLATE",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K",
    "COLLECTION_NAME",
    "EMBEDDING_DIMENSION",
    "DATASET_NAME",
]


def set_dataset(name: str):
    """Switch active dataset and update derived paths."""
    this_module = sys.modules[__name__]
    this_module.DATASET_NAME = name
    this_module.DATASET_DIR = DATA_DIR / name
    this_module.GOLDEN_QA_PATH = this_module.DATASET_DIR / "golden_qa.json"


def load_yaml_config(yaml_path: str | Path) -> dict:
    """Load a YAML config file and override module-level attributes.

    Only keys listed in EXPERIMENT_PARAMS are applied.
    Returns the dict of applied overrides.
    """
    import yaml

    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        print(f"Error: config file not found: {yaml_path}", file=sys.stderr)
        sys.exit(1)

    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}

    this_module = sys.modules[__name__]
    applied = {}
    for key, value in data.items():
        key_upper = key.upper()
        if key_upper in EXPERIMENT_PARAMS:
            setattr(this_module, key_upper, value)
            applied[key_upper] = value

    # If DATASET_NAME was overridden, update derived paths
    if "DATASET_NAME" in applied:
        set_dataset(applied["DATASET_NAME"])

    return applied


def get_param_snapshot() -> dict:
    """Return a dict snapshot of all current experiment parameters."""
    this_module = sys.modules[__name__]
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
    }
    for key in EXPERIMENT_PARAMS:
        snapshot[key] = getattr(this_module, key, None)
    return snapshot
