"""Structured logs: one JSON line per event, every line carries the id of the request it belongs to.

    {"event": "request", "request_id": "7f3a9c1e2b4d", "method": "POST", "path": "/api/ask", "status": 200,
     "duration_ms": 2480, "market": "pl", "question_len": 34, "question_hash": "a3f9c21b7e04", ...}

- `RequestLog` (ASGI middleware, outermost layer) gives every request an id (or reuses a safe X-Request-ID
  from the client), returns it in the X-Request-ID header and writes ONE final "request" line per /api/ call.
- `note(**fields)` adds fields to that final line from anywhere in the code (rag, main, errors).
- `log_event(event, **fields)` writes an extra line right away (e.g. "llm_failed"), with the same id.
- The id lives in a contextvar: every request sees only its own, also inside FastAPI's worker threads
  (they run with a copy of the request's context, and the RequestContext object itself is shared).

Privacy (GDPR: a health question + an IP address can be health data of a real person):
with LOG_QUESTIONS=false (default) the question text is never written; the log gets its length and a keyed
hash (HMAC) instead, so the same question can be recognised without being readable. As a safety net, the
exact question text is replaced by "[question]" if it turns up inside any other logged string (e.g. an error).
"""
import contextvars
import hashlib
import hmac
import json
import re
import time
import uuid

from . import config
from .metrics import metrics

_SAFE_ID = re.compile(r"[A-Za-z0-9_-]{8,64}")

# Events that mean "this answer is worse than it should be" (a step failed or a safety check had to reject the
# answer). They are collected in the request line as "issues": such an answer is not cached (app/cache.py) and
# is counted by the metrics. NOT here: llm_failed (another provider answered, the answer is fine) and
# remote_embed_failed (the local copy of the same model was used, same vectors).
ISSUE_EVENTS = frozenset({
    "translation_failed", "judge_unavailable", "judge_unreadable", "citation_repair_failed",
    "number_repair_failed", "meaning_repair_failed", "uncited_answer_rejected", "wrong_numbers_rejected",
    "wrong_meaning_rejected", "fallback_failed",
})


class RequestContext:
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.fields: dict = {}       # what goes into the final "request" line
        self.question: str = ""      # kept only to redact it from logs (never logged when LOG_QUESTIONS=false)


_ctx: contextvars.ContextVar[RequestContext | None] = contextvars.ContextVar("request_ctx", default=None)


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def current_id() -> str | None:
    ctx = _ctx.get()
    return ctx.request_id if ctx else None


def current_issues() -> list[str]:
    """Issue events (ISSUE_EVENTS) logged so far in the current request."""
    ctx = _ctx.get()
    return list(ctx.fields.get("issues", [])) if ctx else []


def note(**fields) -> bool:
    """Add fields to the current request's final log line. False when there is no request (scripts, tests)."""
    ctx = _ctx.get()
    if ctx is None:
        return False
    ctx.fields.update(fields)
    return True


# ---------------------------------------------------------------- the question: text or fingerprint
def normalize_question(text: str) -> str:
    """Same question = same key: case and extra spaces don't matter ("Ibuprom  z alkoholem?" = "ibuprom z alkoholem?")."""
    return " ".join((text or "").split()).casefold()


def question_hash(text: str, secret: str | None = None) -> str:
    """Keyed fingerprint (HMAC-SHA256) of the normalized question: same text -> same code, the code can't be
    turned back into the text, and without the secret nobody can build a matching code for a guessed question."""
    key = (secret if secret is not None else config.LOG_HASH_SECRET).encode()
    return hmac.new(key, normalize_question(text).encode(), hashlib.sha256).hexdigest()[:12]


def question_fields(text: str) -> dict:
    out = {"question_len": len(text or ""), "question_hash": question_hash(text)}
    if config.LOG_QUESTIONS:
        out["question"] = text
    return out


def note_question(text: str) -> None:
    """Record the current request's question (text only if LOG_QUESTIONS, else length + hash)."""
    ctx = _ctx.get()
    if ctx is not None:
        ctx.question = text or ""
        ctx.fields.update(question_fields(text))


def _redact(value, question: str):
    if not question or config.LOG_QUESTIONS:
        return value
    if isinstance(value, str):
        return value.replace(question, "[question]")
    if isinstance(value, dict):
        return {k: _redact(v, question) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(v, question) for v in value]
    return value


# ---------------------------------------------------------------- writing a line
def log_event(event: str, **fields) -> None:
    """Write one JSON line now. Never raises: a logging problem must not break an answer."""
    ctx = _ctx.get()
    line = {"event": event}
    if ctx is not None:
        line["request_id"] = ctx.request_id
        if event in ISSUE_EVENTS and event not in ctx.fields.get("issues", []):
            ctx.fields["issues"] = [*ctx.fields.get("issues", []), event]
    line.update(fields)
    try:
        if ctx is not None:
            line = _redact(line, ctx.question)
        print(json.dumps(line, default=str), flush=True)
        # The metrics are counted from these same lines (app/metrics.py).
        if event == "request":
            metrics.record_request(line)
        else:
            metrics.record_event(event, line)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------- middleware
class RequestLog:
    """Outermost ASGI layer. Every request: an id + X-Request-ID header. /api/ requests: one final log line."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        sent = dict(scope.get("headers") or []).get(b"x-request-id", b"").decode("latin-1").strip()
        ctx = RequestContext(sent if _SAFE_ID.fullmatch(sent) else new_id())
        token = _ctx.set(ctx)
        t0 = time.perf_counter()
        status = 500  # if the app dies before answering, that's what the client gets

        async def send_with_id(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"x-request-id"]
                headers.append((b"x-request-id", ctx.request_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            if scope["path"].startswith("/api/"):
                fields = {"method": scope["method"], "path": scope["path"], "status": status,
                          "duration_ms": round((time.perf_counter() - t0) * 1000)}
                fields.update({k: v for k, v in ctx.fields.items() if k not in fields})
                log_event("request", **fields)
            _ctx.reset(token)
