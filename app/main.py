"""FastAPI app. Endpoints:
  GET  /api/health   -> ok / degraded (200) or down (503): Qdrant, LLM circuit breaker, embeddings (app/health.py)
  GET  /api/drugs    -> drugs in the knowledge base
  GET  /api/fact     -> short "did you know?" fact for the loading screen
  POST /api/ask      -> {question} -> answer + cited sources
  GET  /api/metrics  -> counters, cache, errors, p50/p95 latency (needs METRICS_TOKEN, app/metrics.py)
  GET  /docs         -> automatic Swagger UI
  GET  /             -> the frontend (served from ./frontend when running locally or on HF)
"""
import hmac
import json
import time

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from . import cache, config, errors, feedback, health, limits, llm, observability, rag, spam
from .metrics import metrics
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
        observability.log_event("warmup_skipped", error=f"{type(e).__name__}: {e}")
    yield


app = FastAPI(title="PharmaRAG", description="Drug-label Q&A with traceable sources", version="1.0",
              lifespan=lifespan)

# Exact HTTP numbers + one JSON shape for every error (see app/errors.py). Must come BEFORE CORS,
# so error answers carry the CORS header too (otherwise the browser hides them).
errors.install(app)

# CORS lets the browser call this API from another domain (the Vercel frontend).
# expose_headers: lets the frontend's JavaScript read X-Request-ID (browsers hide other headers by default).
app.add_middleware(
    CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

# Structured logs: a request id for every request + one JSON log line per /api/ call (app/observability.py).
# Added last = the outermost layer, so it also sees the final status of errors and crashes.
app.add_middleware(observability.RequestLog)


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # unknown field (e.g. a typo "answerLanguage") -> 400, not ignored
    # Checks live in app/inputs.py: the question is cleaned, then 3-500 characters with at least one letter;
    # codes accept "EN" / " pl " and hint at mix-ups ('ua' vs 'uk').
    question: Question = Field(..., examples=["What is metformin used for?"])
    top_k: int | None = Field(None, ge=1, le=15)
    answer_language: AnswerLanguage = "auto"
    market: Market = "us"


@app.get("/api/health")
def health_check():
    # Touches Qdrant on purpose: a daily ping (GitHub Actions) keeps the free cluster from being suspended.
    # "down" = questions can't be answered -> 503, so `curl -f` in the keepalive fails and GitHub e-mails the owner.
    report = health.report()
    observability.note(health=report["status"])
    if report["status"] == "down":
        broken = ", ".join(name for name, c in report["checks"].items() if c["status"] == "down")
        error = errors.error_response(503, "UNAVAILABLE", f"Not able to answer questions right now ({broken} down).",
                                      headers={"Retry-After": str(round(config.HEALTH_CACHE_S))})
        # the usual error fields (detail, error, request_id) + the full report, so the keepalive log shows what broke
        return JSONResponse({**json.loads(error.body), **report}, status_code=503,
                            headers={"Retry-After": error.headers["retry-after"]})
    return report


# "Bearer" security: the token goes in the header "Authorization: Bearer <token>", never in the address (addresses
# end up in browser history and logs). Declaring it this way also gives Swagger (/docs) its "Authorize" button.
_metrics_token = HTTPBearer(auto_error=False, scheme_name="METRICS_TOKEN",
                            description="The METRICS_TOKEN from the server settings")


@app.get("/api/metrics")
def get_metrics(creds: HTTPAuthorizationCredentials | None = Depends(_metrics_token)):
    if not config.METRICS_TOKEN:  # not set up = switched off
        return errors.error_response(404, "NOT_FOUND", "Not Found")
    sent = creds.credentials if creds else ""
    # compare_digest takes the same time however many characters match, so the token can't be guessed letter by letter
    if not hmac.compare_digest(sent.encode(), config.METRICS_TOKEN.encode()):
        raise errors.ApiError(401, "UNAUTHORIZED", "A valid metrics token is required (Authorization: Bearer ...).",
                              headers={"WWW-Authenticate": "Bearer"})
    return metrics.snapshot()


@app.get("/api/drugs")
def drugs(market: Market = "us"):
    return rag.library(market)


@app.get("/api/fact")
def fact(seen: str = "", lang: UiLanguage = "en"):
    """A short curated fact for the loading screen. `seen` = drugs already shown (comma-separated)."""
    return rag.random_fact({d for d in seen.split(",") if d}, lang)


@app.post("/api/ask")
def ask(req: AskRequest, request: Request):
    # What the final log line knows about this question (text only if LOG_QUESTIONS, else length + hash).
    observability.note_question(req.question)
    observability.note(market=req.market, answer_language=req.answer_language, top_k=req.top_k)
    limits.limit_ask(request)  # 429 after ASK_LIMIT_PER_MINUTE / ASK_LIMIT_PER_DAY questions (app/limits.py)

    # Same question (same country, answer language, top_k) in the last 15 min: the saved answer (app/cache.py).
    t0 = time.perf_counter()
    key = cache.answers.key(req.question, req.market, req.answer_language, req.top_k)
    saved = cache.answers.get(key)
    if saved is not None:
        took = round((time.perf_counter() - t0) * 1000)
        observability.note(cache="hit", timings={"total_ms": took},
                           **{k: saved[k] for k in _CACHED_LOG_FIELDS if k in saved})
        # The old stage timings would be misleading (nothing was searched or generated): only the lookup time.
        return {**saved, "cached": True, "timings": {"total_ms": took}, "request_id": observability.current_id()}
    cache_state = {"cache": "skip", "cache_skip_reason": "error"} if cache.answers.enabled() else {"cache": "off"}

    try:
        result = rag.ask(req.question, req.top_k, req.answer_language, req.market)
    except llm.LLMUnavailable as e:
        # Every AI provider is down/busy: a clean 503 (the UI shows "busy, try again" + a retry button).
        # The raw provider errors go to the log only.
        observability.note(**cache_state)  # errors are never cached
        observability.log_event("llm_unavailable", error=str(e))
        raise errors.ApiError(503, "LLM_UNAVAILABLE", headers={"Retry-After": str(e.retry_after_s)},
                              detail=f"LLM_UNAVAILABLE: the AI service is busy right now. "
                                     f"Please try again in {e.retry_after_s} s.") from e
    except Exception as e:  # Qdrant / embeddings / LLM failed: details to the log only, never to the user
        observability.note(**cache_state)
        observability.log_event("ask_failed", error=f"{type(e).__name__}: {e}")
        if errors.looks_busy(e):  # "503 overloaded", "429 too many": busy, not broken -> the UI offers a retry
            raise errors.ApiError(503, "SERVICE_BUSY", headers={"Retry-After": "10"},
                                  detail="SERVICE_BUSY (503): a service we use is busy right now. "
                                         "Please try again in 10 s.") from e
        raise errors.ApiError(502, "UPSTREAM_ERROR", "The search service is not responding right now. "
                                                     "Please try again in a moment.") from e
    if not cache.answers.enabled():
        observability.note(cache="off")
    elif reason := cache.skip_reason(result, observability.current_issues()):
        # emergency / LLM down / unchecked meaning / an issue: not saved, the next asker gets a fresh try
        observability.note(cache="skip", cache_skip_reason=reason)
    else:
        cache.answers.put(key, result)
        observability.note(cache="miss")
    return {**result, "cached": False, "request_id": observability.current_id()}


# What a cache hit writes into its log line (the rest, e.g. risk or issues, belongs to the original answer).
_CACHED_LOG_FIELDS = ("language", "answer_language", "detected_drugs", "found", "mode", "model", "meaning_check")


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
        observability.log_event("feedback_failed", error=f"{type(e).__name__}: {e}")
        raise errors.ApiError(502, "UPSTREAM_ERROR", "Could not send the report right now. "
                                                     "Please try again later.") from e
    spam.remember_sent(visitor, req.message)  # only e-mails really sent count (daily total, same-text check)
    return {"ok": True}


# Serve the static frontend at "/" (must be mounted last so it doesn't shadow /api routes).
_frontend = config.ROOT / "frontend"
if _frontend.exists():
    app.mount("/", StaticFiles(directory=_frontend, html=True), name="frontend")
