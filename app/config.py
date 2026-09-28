"""Central configuration. Everything is read from environment variables (.env locally,
"Secrets" on Hugging Face Spaces), so the same code runs locally and in the cloud."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- LLM -------------------------------------------------------------------
# Comma-separated, in order of preference: the first provider that answers wins.
# e.g. "groq,gemini" = fast Groq first, Gemini as a backup.  Options: groq | gemini | anthropic
LLM_PROVIDERS = [p.strip().lower() for p in os.getenv("LLM_PROVIDER", "gemini").split(",") if p.strip()]
LLM_PROVIDER = ",".join(LLM_PROVIDERS)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
# Optional comma-separated backup models, used if the main one keeps failing (e.g. overloaded).
# How much the model "thinks" before answering: minimal | low | medium | high. Grounded Q&A over
# provided sources doesn't need deep reasoning, and thinking is the biggest source of latency.
GEMINI_THINKING = os.getenv("GEMINI_THINKING", "low").upper()
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "30"))  # per LLM call; avoids 2-minute hangs
GEMINI_FALLBACK_MODELS = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "").split(",") if m.strip()]
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")

# --- Embeddings (run locally with fastembed, no API key, no cost) ----------
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")  # 384-dim, English (US labels)
# Polish / Ukrainian leaflets need a multilingual model; it also lets a question in one language
# match a leaflet written in another (e.g. a Ukrainian question over Polish leaflets).
MULTI_EMBED_MODEL = os.getenv("MULTI_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

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
MIN_SCORE_MULTI = float(os.getenv("MIN_SCORE_MULTI", "0.35"))  # the multilingual model scores lower overall

# --- Markets ------------------------------------------------------------------
US_FULL = (ROOT / "data" / "markets" / "us.json").exists()
# Each market = one Qdrant collection built from that country's official register.
MARKETS = {
    # Full US register (scripts/fetch_us.py) once it's indexed; the 44-drug starter collection before that.
    "us": {"collection": "labels_us" if US_FULL else COLLECTION, "model": EMBED_MODEL, "min_score": MIN_SCORE,
           "source": "FDA"},
    "pl": {"collection": "leaflets_pl", "model": MULTI_EMBED_MODEL, "min_score": MIN_SCORE_MULTI, "source": "URPL + EMA"},
    "ua": {"collection": "leaflets_ua", "model": MULTI_EMBED_MODEL, "min_score": MIN_SCORE_MULTI, "source": "MOZ"},
    "cz": {"collection": "leaflets_cz", "model": MULTI_EMBED_MODEL, "min_score": MIN_SCORE_MULTI, "source": "SÚKL"},
    "es": {"collection": "leaflets_es", "model": MULTI_EMBED_MODEL, "min_score": MIN_SCORE_MULTI, "source": "AEMPS"},
    "fr": {"collection": "leaflets_fr", "model": MULTI_EMBED_MODEL, "min_score": MIN_SCORE_MULTI, "source": "ANSM"},
}

# --- Feedback ("Report a problem" form) ------------------------------------------
# Messages are e-mailed through Resend (free tier). The address lives only here on the server,
# so visitors never see it.
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
FEEDBACK_TO = os.getenv("FEEDBACK_TO", "")

# --- Web ----------------------------------------------------------------------
# Comma-separated list of frontend origins allowed to call the API (e.g. your Vercel URL).
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

DATA_FILE = ROOT / "data" / "labels.jsonl"
