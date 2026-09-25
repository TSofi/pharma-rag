"""Central configuration. Everything is read from environment variables (.env locally,
"Secrets" on Hugging Face Spaces), so the same code runs locally and in the cloud."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- LLM -------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()  # "gemini" | "anthropic"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
# Optional comma-separated backup models, used if the main one keeps failing (e.g. overloaded).
GEMINI_FALLBACK_MODELS = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "").split(",") if m.strip()]
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")

# --- Embeddings (run locally with fastembed, no API key, no cost) ----------
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")  # 384-dim

# --- Vector database (Qdrant) ----------------------------------------------
# If QDRANT_URL is empty we use Qdrant's embedded "local mode" (a folder on disk),
# so local development needs no account. In production we point at Qdrant Cloud.
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", str(ROOT / "qdrant_data"))
COLLECTION = os.getenv("QDRANT_COLLECTION", "drug_labels")

# --- Retrieval --------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "6"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.62"))  # below this a chunk is "not relevant"

# --- Web ----------------------------------------------------------------------
# Comma-separated list of frontend origins allowed to call the API (e.g. your Vercel URL).
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

DATA_FILE = ROOT / "data" / "labels.jsonl"
