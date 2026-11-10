"""Central configuration. Everything is read from environment variables (.env locally,
"Secrets" on Hugging Face Spaces), so the same code runs locally and in the cloud."""
import os
import secrets
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
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "30"))  # per Gemini/Anthropic call; avoids 2-minute hangs
GROQ_TIMEOUT_S = float(os.getenv("GROQ_TIMEOUT_S", "12"))  # Groq normally answers in 1-2 s; 12 s = it's stuck
LLM_DEADLINE_S = float(os.getenv("LLM_DEADLINE_S", "30"))  # all LLM calls of ONE question together, then "busy"
# Meaning check: a second LLM call (the "judge") checks that each cited sentence says what its source says.
# risky = only when the question/answer touches pregnancy, breastfeeding, children, alcohol, driving, other
# medicines, allergy, kidney/liver | always | off
JUDGE_MODE = os.getenv("JUDGE_MODE", "risky").strip().lower()
# true = a sentence the source doesn't contain (NOT_IN_SOURCE) is a problem like a contradiction
JUDGE_STRICT = os.getenv("JUDGE_STRICT", "true").strip().lower() in ("1", "true", "yes", "on")
GEMINI_FALLBACK_MODELS = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "").split(",") if m.strip()]
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")

# --- Embeddings (run locally with fastembed, no API key, no cost) ----------
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")  # 384-dim, English (US labels)
# Polish / Ukrainian leaflets need a multilingual model; it also lets a question in one language
# match a leaflet written in another (e.g. a Ukrainian question over Polish leaflets).
# Models whose QUESTION embeddings are computed by the Hugging Face Inference API instead of locally
# (set on the 512 MB server). Leaflets were embedded locally with the same model when indexing.
HF_TOKEN = os.getenv("HF_TOKEN", "")
REMOTE_EMBED_MODELS = [m.strip() for m in os.getenv("REMOTE_EMBED_MODELS", "").split(",") if m.strip()]
# How many embedding models to keep loaded at once (1 on the free 512 MB server, 2 if RAM allows).
MAX_MODELS_IN_MEMORY = int(os.getenv("MAX_MODELS_IN_MEMORY", "1"))
EMBED_THREADS = int(os.getenv("EMBED_THREADS", "0"))  # 0 = let ONNX Runtime decide
MULTI_EMBED_MODEL = os.getenv("MULTI_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# int8 copies of both models (app/int8_models.py); used instead of the originals when present.
INT8_MODEL_DIR = os.getenv("INT8_MODEL_DIR", str(ROOT / "models_int8"))

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
# Questions per visitor (app/limits.py); 0 = off (default: off until the client IP behind Render's proxy is confirmed; suggested 10 / 200). Each question costs up to 4-5 LLM calls on the free keys.
ASK_LIMIT_PER_MINUTE = int(os.getenv("ASK_LIMIT_PER_MINUTE", "0"))
ASK_LIMIT_PER_DAY = int(os.getenv("ASK_LIMIT_PER_DAY", "0"))
# "Report a problem" form (app/spam.py); 0 = off. Resend's free plan: 100 e-mails a day, 3,000 a month.
FEEDBACK_LIMIT_PER_HOUR = int(os.getenv("FEEDBACK_LIMIT_PER_HOUR", "5"))   # per visitor
FEEDBACK_LIMIT_PER_DAY = int(os.getenv("FEEDBACK_LIMIT_PER_DAY", "90"))    # all visitors together

# --- Logs (app/observability.py) ----------------------------------------------------
# A health question + an IP address can be health data of a real person (GDPR art. 9), so by default the
# question TEXT is not logged: only its length and a keyed fingerprint (hash). true = log the text (debugging).
LOG_QUESTIONS = os.getenv("LOG_QUESTIONS", "false").strip().lower() in ("1", "true", "yes", "on")
# Key for the question fingerprint. Set it on the server so fingerprints stay comparable across restarts;
# empty = a random key per start (fingerprints then only match within one run).
LOG_HASH_SECRET = os.getenv("LOG_HASH_SECRET", "").strip() or secrets.token_hex(16)

# --- Health check (app/health.py) --------------------------------------------------------
# The result of /api/health is reused for this many seconds (the website calls it on every page load).
HEALTH_CACHE_S = float(os.getenv("HEALTH_CACHE_S", "15"))
# How long /api/health waits for Qdrant before calling it "down" (normal questions still wait up to 60 s).
HEALTH_QDRANT_TIMEOUT_S = float(os.getenv("HEALTH_QDRANT_TIMEOUT_S", "3"))

# --- Answer cache (app/cache.py) -----------------------------------------------------------
# The same question (same country, answer language, top_k) within this many seconds gets the saved answer:
# no Qdrant / LLM call. Only clean answers are saved (no emergency, no errors, no issues). 0 = cache off.
ANSWER_CACHE_TTL_S = float(os.getenv("ANSWER_CACHE_TTL_S", "900"))
ANSWER_CACHE_MAX = int(os.getenv("ANSWER_CACHE_MAX", "300"))  # answers kept at most (~10 KB each, in RAM only)

# --- Metrics (app/metrics.py, GET /api/metrics) -----------------------------------------------
# Secret token for /api/metrics (header "Authorization: Bearer <token>"). Empty = the endpoint is switched off (404).
METRICS_TOKEN = os.getenv("METRICS_TOKEN", "").strip()
METRICS_WINDOW = int(os.getenv("METRICS_WINDOW", "1000"))  # p50/p95 over the last N answers per stage

DATA_FILE = ROOT / "data" / "labels.jsonl"
