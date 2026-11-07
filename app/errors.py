"""One shape and an exact HTTP number for every error the API returns.

    {"detail": "<short readable text>", "error": "<CODE>", "field": "<which field>" (400 only)}

    400 the request is broken or wrong (bad JSON, missing/too long/wrong field)   -> the user can fix it
    404 unknown address            405 right address, wrong method (GET instead of POST)
    413 the request is too big     429 too many requests, wait (Retry-After header)
    500 a bug on our side          502 a service we depend on failed (Qdrant, LLM, e-mail)
    503 busy / not set up, try again later

Rules: `detail` is always a string (the frontend shows it), the user's input is never sent back, and
internal details (exception text, URLs, keys) go to the log only.
`install(app)` wires everything in; call it BEFORE adding CORS so error answers still carry CORS headers.
"""
import json
import re
import traceback
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.routing import Route

# Biggest request body we accept, in bytes. A question is <= 500 characters; a report can carry a
# screenshot of up to 4 MB, which is ~5.6 MB as base64 text, plus message and context.
MAX_BODY_DEFAULT = 16 * 1024
MAX_BODY = {"/api/feedback": 6_500_000}

# Default code for each HTTP number (used when the code that raised the error didn't choose one).
_CODES = {400: "BAD_REQUEST", 404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED", 413: "TOO_LARGE",
          429: "RATE_LIMITED", 500: "INTERNAL_ERROR", 502: "UPSTREAM_ERROR", 503: "UNAVAILABLE"}


# Words a busy/overloaded service puts in its error (Gemini, Groq, Qdrant). Same list the frontend uses to
# show "busy, try again" instead of "something went wrong".
_BUSY = re.compile(r"\b(503|429)\b|UNAVAILABLE|overload|high demand|RESOURCE_EXHAUSTED", re.I)


def looks_busy(exc: Exception) -> bool:
    return bool(_BUSY.search(str(exc)))


class ApiError(StarletteHTTPException):
    """Raise this in an endpoint: ApiError(429, "RATE_LIMITED", "Too many ...", headers={"Retry-After": "60"})."""
    def __init__(self, status_code: int, code: str, detail: str, headers: dict | None = None):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.code = code


def error_response(status: int, code: str, detail: str, headers: dict | None = None, **extra) -> JSONResponse:
    return JSONResponse({"detail": detail, "error": code, **extra}, status_code=status, headers=headers)


# ---------- 400: turn FastAPI's validation errors (422 + a list) into one readable message ----------

def _field(loc) -> str:
    # loc looks like ("body", "question") or ("query", "lang"): drop where it came from, keep the name.
    return ".".join(str(p) for p in loc[1:]) or str(loc[0])


def describe(err: dict) -> tuple[str, str, str | None]:
    """One pydantic error -> (code, readable message, field). Never includes the input itself."""
    kind, loc, ctx = err.get("type", ""), tuple(err.get("loc", ())), err.get("ctx") or {}
    if kind == "json_invalid":
        return "INVALID_JSON", "The request body is not valid JSON.", None
    if len(loc) <= 1 and loc[:1] == ("body",):  # the body as a whole is missing or not an object
        return ("INVALID_BODY", "The request body must be a JSON object, for example "
                                '{"question": "What is metformin used for?"} (Content-Type: application/json).', None)
    name = _field(loc)
    if kind.startswith("app_"):  # our own checks (app/inputs.py): the message is already readable
        return kind[4:].upper(), f"'{name}' {err.get('msg')}", name
    if kind == "missing":
        return "MISSING_FIELD", f"'{name}' is required.", name
    if kind == "string_too_short":
        return "TOO_SHORT", f"'{name}' is too short (at least {ctx.get('min_length')} characters).", name
    if kind == "string_too_long":
        return "TOO_LONG", f"'{name}' is too long (at most {ctx.get('max_length')} characters).", name
    if kind == "string_type":
        return "WRONG_TYPE", f"'{name}' must be text.", name
    if kind in ("int_type", "int_parsing", "int_from_float"):
        return "WRONG_TYPE", f"'{name}' must be a whole number.", name
    if kind in ("greater_than_equal", "greater_than"):
        return "OUT_OF_RANGE", f"'{name}' must be at least {ctx.get('ge', ctx.get('gt'))}.", name
    if kind in ("less_than_equal", "less_than"):
        return "OUT_OF_RANGE", f"'{name}' must be at most {ctx.get('le', ctx.get('lt'))}.", name
    if kind == "literal_error":
        return "INVALID_VALUE", f"'{name}' must be one of: {ctx.get('expected')}.", name
    if kind == "extra_forbidden":
        return "UNKNOWN_FIELD", f"'{name}' is not a known field.", name
    return "INVALID_VALUE", f"'{name}' is not valid: {err.get('msg', 'wrong value')}.", name


async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    code, detail, field = describe(exc.errors()[0])  # the first problem is enough to fix the request
    return error_response(400, code, detail, field=field)


# ---------- 404 / 405 / our own ApiError / any other HTTPException ----------

def _allowed_methods(request: Request) -> set[str]:
    """Methods of the API routes whose address matches this path (empty = the address doesn't exist)."""
    path = request.url.path
    return {m for r in request.app.routes if isinstance(r, Route) and r.path_regex.match(path)
            for m in (r.methods or ())}


async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    status, headers = exc.status_code, dict(exc.headers or {})
    detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
    # The frontend is mounted at "/", so a request with the wrong method (GET /api/ask) falls through to
    # it and comes back as 404 or "405, use GET". For /api/... we decide from the API routes instead:
    # the address exists with another method -> 405 naming the right one; it doesn't exist -> 404.
    if status in (404, 405) and request.url.path.startswith("/api/"):
        allowed = _allowed_methods(request) - {"HEAD"}
        headers.pop("allow", None), headers.pop("Allow", None)
        if allowed:
            status, detail = 405, f"Use {' or '.join(sorted(allowed))} for this address."
            headers["Allow"] = ", ".join(sorted(allowed))
        else:
            status, detail = 404, "Not Found"
    code = getattr(exc, "code", None) if status == exc.status_code else None
    return error_response(status, code or _CODES.get(status, "ERROR"), detail, headers or None)


# ---------- 413 and 500: two small ASGI layers around the app ----------

class BodySizeLimit:
    """Refuses a body bigger than the limit with 413 BEFORE the app reads or parses it.
    Checks the Content-Length header and also counts the bytes (a client can stream without that header)."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH"):
            return await self.app(scope, receive, send)
        limit = MAX_BODY.get(scope["path"], MAX_BODY_DEFAULT)
        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > limit:
            return await _too_large(limit)(scope, receive, send)
        body, size, ended = [], 0, None
        while True:  # read the body ourselves, counting, then hand it to the app unchanged
            message = await receive()
            if message["type"] != "http.request":  # the client went away
                ended = message
                break
            size += len(message.get("body", b""))
            if size > limit:
                return await _too_large(limit)(scope, receive, send)
            body.append(message.get("body", b""))
            if not message.get("more_body"):
                break
        replay = [ended or {"type": "http.request", "body": b"".join(body), "more_body": False}]

        async def replay_receive():
            return replay.pop() if replay else await receive()
        return await self.app(scope, replay_receive, send)


def _too_large(limit: int) -> JSONResponse:
    size = f"{limit / 1_000_000:.1f} MB" if limit >= 1_000_000 else f"{limit // 1024} KB"
    return error_response(413, "TOO_LARGE", f"The request is too large (limit {size}).")


class CatchCrashes:
    """Any unexpected exception -> JSON 500 with a short id; the full error + id go to the log."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        started = False

        async def watch_send(message):
            nonlocal started
            started = started or message["type"] == "http.response.start"
            await send(message)
        try:
            await self.app(scope, receive, watch_send)
        except Exception:  # noqa: BLE001 -- that's the point: nothing unexpected reaches the user raw
            request_id = uuid.uuid4().hex[:12]
            print(f"[api] error {request_id} on {scope['method']} {scope['path']}\n{traceback.format_exc()}",
                  flush=True)
            if started:  # half an answer is already on its way: nothing better we can do
                raise
            await error_response(500, "INTERNAL_ERROR",
                                 f"Something went wrong on our side. Please try again later (error id {request_id}).",
                                 request_id=request_id)(scope, receive, send)


def install(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_middleware(BodySizeLimit)   # inner
    app.add_middleware(CatchCrashes)    # outer (catches a crash in BodySizeLimit too)
