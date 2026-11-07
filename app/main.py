"""FastAPI app. Endpoints:
  GET  /api/health   -> is the service up, which LLM/vector DB are used
  GET  /api/drugs    -> drugs in the knowledge base
  GET  /api/fact     -> short "did you know?" fact for the loading screen
  POST /api/ask      -> {question} -> answer + cited sources
  GET  /docs         -> automatic Swagger UI
  GET  /             -> the frontend (served from ./frontend when running locally or on HF)
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from . import config, errors, feedback, limits, llm, rag, spam
from .inputs import AnswerLanguage, Contact, Market, Question, ReportMessage, UiLanguage

from contextlib import asynccontextmanager

from . import store


@asynccontextmanager
async def lifespan(_app):
    # Warm-up: load the embedding model and open the Qdrant connection at startup,
    # so the FIRST user question doesn't pay for it (a few seconds on a small CPU).
    try:
        store.get_client()
        # Warm up whichever model runs locally (a model served by the embedding API needs no RAM here).
        local = [m for m in (config.MULTI_EMBED_MODEL, config.EMBED_MODEL)
                 if not (m in config.REMOTE_EMBED_MODELS and config.HF_TOKEN)]
        if local:
            store.embed_query("warm up", local[0])
        for m in config.MARKETS:
            rag.markets.index(m)  # load the product indexes (~1 s) before the first question
    except Exception as e:  # noqa: BLE001 -- never block startup on warm-up
        print(f"warm-up skipped: {e}", flush=True)
    yield


app = FastAPI(title="PharmaRAG", description="Drug-label Q&A with traceable sources", version="1.0",
              lifespan=lifespan)

# Exact HTTP numbers + one JSON shape for every error (see app/errors.py). Must come BEFORE CORS,
# so error answers carry the CORS header too (otherwise the browser hides them).
errors.install(app)

# CORS lets the browser call this API from another domain (the Vercel frontend).
app.add_middleware(
    CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"]
)


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # unknown field (e.g. a typo "answerLanguage") -> 400, not ignored
    # Checks live in app/inputs.py: the question is cleaned, then 3-500 characters with at least one letter;
    # codes accept "EN" / " pl " and hint at mix-ups ('ua' vs 'uk').
    question: Question = Field(..., examples=["What is metformin used for?"])
    top_k: int | None = Field(None, ge=1, le=15)
    answer_language: AnswerLanguage = "auto"
    market: Market = "us"


@app.get("/api/health")
def health():
    # Touches Qdrant on purpose: a daily ping (GitHub Actions) keeps the free cluster from being suspended.
    try:
        qdrant = len(store.get_client().get_collections().collections)
    except Exception as e:  # noqa: BLE001
        qdrant = f"error: {type(e).__name__}"
    return {
        "status": "ok",
        "qdrant_collections": qdrant,
        "llm": config.LLM_PROVIDER,
        "vector_db": "qdrant-cloud" if config.QDRANT_URL else "qdrant-local",
        "drugs": len(rag.library()),
        "markets": {m: len(rag.library(m)) for m in config.MARKETS},
    }


@app.get("/api/drugs")
def drugs(market: Market = "us"):
    return rag.library(market)


@app.get("/api/fact")
def fact(seen: str = "", lang: UiLanguage = "en"):
    """A short curated fact for the loading screen. `seen` = drugs already shown (comma-separated)."""
    return rag.random_fact({d for d in seen.split(",") if d}, lang)


@app.post("/api/ask")
def ask(req: AskRequest, request: Request):
    limits.limit_ask(request)  # 429 after ASK_LIMIT_PER_MINUTE / ASK_LIMIT_PER_DAY questions (app/limits.py)
    try:
        return rag.ask(req.question, req.top_k, req.answer_language, req.market)
    except llm.LLMUnavailable as e:
        # Every AI provider is down/busy: a clean 503 (the UI shows "busy, try again" + a retry button).
        # The raw provider errors go to the log only.
        print(f"[api] LLM unavailable: {e}", flush=True)
        raise errors.ApiError(503, "LLM_UNAVAILABLE", headers={"Retry-After": str(e.retry_after_s)},
                              detail=f"LLM_UNAVAILABLE: the AI service is busy right now. "
                                     f"Please try again in {e.retry_after_s} s.") from e
    except Exception as e:  # Qdrant / embeddings / LLM failed: details to the log only, never to the user
        print(f"[api] ask failed: {type(e).__name__}: {e}", flush=True)
        if errors.looks_busy(e):  # "503 overloaded", "429 too many": busy, not broken -> the UI offers a retry
            raise errors.ApiError(503, "SERVICE_BUSY", headers={"Retry-After": "10"},
                                  detail="SERVICE_BUSY (503): a service we use is busy right now. "
                                         "Please try again in 10 s.") from e
        raise errors.ApiError(502, "UPSTREAM_ERROR", "The search service is not responding right now. "
                                                     "Please try again in a moment.") from e


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # same: only the fields below (app.js sends exactly these)
    message: ReportMessage                           # cleaned (line breaks kept), 5-3000, letters, <= 2 links
    contact: Contact = Field("", max_length=200)     # one line: line breaks / hidden characters removed
    context: str = Field("", max_length=6000)       # last question/answer, market, browser (if the user agrees)
    image: str | None = Field(None, max_length=6_000_000)  # screenshot as a data: URL
    website: str = ""                              # honeypot: real people never fill this hidden field
    form_ms: int | None = Field(None, ge=0)        # how long the form was open (app.js); < 3 s or missing = a bot


@app.post("/api/feedback")
def send_feedback(req: FeedbackRequest, request: Request):
    if not (config.RESEND_API_KEY and config.FEEDBACK_TO):  # our setup is missing, not the visitor's fault
        raise errors.ApiError(503, "FEEDBACK_UNAVAILABLE", "Reporting is not available right now.")
    # The real visitor (last address in the proxy note), not Render's proxy: otherwise everyone shares one counter.
    ip = limits.client_ip(request)
    visitor = limits.visitor_key(ip)
    reason = spam.bot_reason(req.website, req.form_ms, visitor, req.message)  # app/spam.py
    if reason:  # hidden field filled / sent too fast / same text again: pretend it worked, the bot learns nothing
        spam.log_drop(reason, ip)
        return {"ok": True}
    spam.check_limits(visitor)  # 429: too many from this visitor this hour, or the day's total reached
    try:
        feedback.send(req.message, req.contact, req.context, req.image)
    except ValueError as e:  # only the screenshot check raises this; the text is ours, safe to show
        raise errors.ApiError(400, "INVALID_IMAGE", str(e)) from e
    except Exception as e:  # noqa: BLE001 -- mail service down: details to the log only
        print(f"[api] feedback failed: {type(e).__name__}: {e}", flush=True)
        raise errors.ApiError(502, "UPSTREAM_ERROR", "Could not send the report right now. "
                                                     "Please try again later.") from e
    spam.remember_sent(visitor, req.message)  # only e-mails really sent count (daily total, same-text check)
    return {"ok": True}


# Serve the static frontend at "/" (must be mounted last so it doesn't shadow /api routes).
_frontend = config.ROOT / "frontend"
if _frontend.exists():
    app.mount("/", StaticFiles(directory=_frontend, html=True), name="frontend")
