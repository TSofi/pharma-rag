"""Thin wrapper around the LLM API. Swapping providers = changing LLM_PROVIDER in .env.

Reliability rules (tests/test_llm.py):
- providers are tried in order (e.g. groq -> gemini -> gemini backups); the first real answer wins;
- one Groq try waits at most GROQ_TIMEOUT_S, and all tries of one question together at most LLM_DEADLINE_S;
- an empty, safety-blocked or cut-off answer counts as a failure, so the next provider is asked;
- a provider that is overloaded / timing out is skipped for COOLDOWN_S (circuit breaker), but a bad
  request (e.g. a too-long question) does not bench it for everyone;
- if every provider is benched, one "probe" is still sent every PROBE_INTERVAL_S, so the site comes
  back as soon as a provider does;
- if nothing answers, LLMUnavailable is raised (the API turns it into 503 "busy, try again").
"""
import contextlib
import contextvars
import re
import time

from . import config, observability


class LLMUnavailable(RuntimeError):
    """No provider could answer (all failed, all benched, or the time budget ran out)."""
    def __init__(self, message: str, retry_after_s: int = 10):
        super().__init__(message)
        self.retry_after_s = retry_after_s


class EmptyAnswer(RuntimeError):
    """The provider replied, but with nothing usable: empty, blocked by a safety filter, or cut off."""


# ---------------------------------------------------------------- error classification
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
RETRYABLE_WORDS = ("UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded")
REQUEST_ERROR_STATUS = {400, 404, 413, 422}  # a problem with THIS request, not with the provider
_MODEL_GONE = re.compile(r"model_not_found|model_decommissioned|decommissioned|does not exist", re.I)


def _status(e: Exception) -> int | None:
    """HTTP status of an error: from the SDK attribute, else the leading number ("503 Service..."),
    else a stand-alone 429/5xx in the text. Never a number hidden inside another ("15004 tokens")."""
    for attr in ("status_code", "code", "status"):
        v = getattr(e, attr, None)
        if isinstance(v, int) and 100 <= v <= 599:
            return v
    m = re.match(r"\s*(\d{3})\b", str(e)) or re.search(r"\b(429|500|502|503|504)\b", str(e))
    return int(m.group(1)) if m else None


def _is_retryable(e: Exception) -> bool:
    status = _status(e)
    if status is not None:
        return status in RETRYABLE_STATUS
    return any(w in str(e) for w in RETRYABLE_WORDS)


def _is_request_error(e: Exception) -> bool:
    """Errors caused by this request (or this answer), which say nothing about the provider's health.
    A missing/retired model is NOT one of them: it would fail every request, so it benches the provider."""
    if isinstance(e, EmptyAnswer):
        return True
    return _status(e) in REQUEST_ERROR_STATUS and not _MODEL_GONE.search(str(e))


# ---------------------------------------------------------------- time budget + per-request info
MIN_ATTEMPT_S = 3.0  # don't start a call with less time than this left


class _Scope:
    def __init__(self):
        self.deadline: float | None = None  # set by the first LLM call of the request
        self.model = ""                     # model that produced the last answer of THIS request


_scope: contextvars.ContextVar[_Scope | None] = contextvars.ContextVar("llm_scope", default=None)


@contextlib.contextmanager
def request_scope():
    """One user question = one scope: all its LLM calls share one LLM_DEADLINE_S budget, and
    model_used() reports only models used for this question (FastAPI serves requests in threads, so a
    module global would mix users up)."""
    token = _scope.set(_Scope())
    try:
        yield
    finally:
        _scope.reset(token)


def _deadline() -> float:
    scope = _scope.get()
    if scope is None:  # called outside a request (scripts, tests): one budget per call
        return time.time() + config.LLM_DEADLINE_S
    if scope.deadline is None:
        scope.deadline = time.time() + config.LLM_DEADLINE_S
    return scope.deadline


def model_used() -> str:
    scope = _scope.get()
    return scope.model if scope is not None else last_model_used


def _record_model(name: str) -> None:
    global last_model_used
    last_model_used = name
    scope = _scope.get()
    if scope is not None:
        scope.model = name


def strip_reasoning(text: str) -> str:
    """Reasoning models (deepseek-r1, qwen3...) put their private notes in <think>...</think>."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
    return re.sub(r"<think>.*", "", text, flags=re.S | re.I).strip()  # unclosed: all of it is notes


def _with_retries(fn, attempts: int = 4, deadline: float | None = None):
    """Free-tier APIs are sometimes overloaded (HTTP 503) or rate-limited (429).
    Retry with exponential backoff: wait 1s, 2s, 4s between attempts (never past the deadline)."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            wait = 2 ** i
            out_of_time = deadline is not None and time.time() + wait + MIN_ATTEMPT_S > deadline
            if i == attempts - 1 or not _is_retryable(e) or out_of_time:
                raise
            time.sleep(wait)


# ---------------------------------------------------------------- Gemini
_thinking_supported: dict[str, bool] = {}


def _gemini(model: str, system: str, user: str, timeout: float | None = None) -> str:
    from google import genai
    from google.genai import types

    seconds = timeout or config.LLM_TIMEOUT_S
    client = genai.Client(api_key=config.GEMINI_API_KEY, http_options=types.HttpOptions(timeout=int(seconds * 1000)))
    cfg = dict(system_instruction=system, temperature=0, max_output_tokens=2048)
    if _thinking_supported.get(model, True) and config.GEMINI_THINKING:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_level=config.GEMINI_THINKING)
    try:
        resp = client.models.generate_content(model=model, contents=user,
                                              config=types.GenerateContentConfig(**cfg))
    except Exception as e:  # noqa: BLE001
        # Some (older/lite) models don't accept a thinking level: remember that and retry without it.
        if "thinking" in str(e).lower() and "thinking_config" in cfg:
            _thinking_supported[model] = False
            cfg.pop("thinking_config")
            resp = client.models.generate_content(model=model, contents=user,
                                                  config=types.GenerateContentConfig(**cfg))
        else:
            raise
    finish = str(getattr((getattr(resp, "candidates", None) or [None])[0], "finish_reason", "") or "")
    if "MAX_TOKENS" in finish:
        raise EmptyAnswer(f"gemini {model}: answer cut off (finish_reason={finish})")
    if not (resp.text or "").strip():
        raise EmptyAnswer(f"gemini {model}: empty answer (blocked by a safety filter? finish_reason={finish})")
    return resp.text


def _anthropic(system: str, user: str, timeout: float | None = None) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY, timeout=timeout or config.LLM_TIMEOUT_S)
    msg = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=800,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if msg.stop_reason == "max_tokens":
        raise EmptyAnswer("anthropic: answer cut off (max_tokens)")
    return "".join(b.text for b in msg.content if b.type == "text")


_discovered: list[str] | None = None
last_model_used: str = ""  # last model that answered (any request); use model_used() for THIS request


def _discover_backup_models() -> list[str]:
    """If no GEMINI_FALLBACK_MODELS are configured, ask the API which other 'flash' models this key
    can use (lighter/older ones are less often overloaded). Cached after the first call."""
    global _discovered
    if _discovered is None:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=config.GEMINI_API_KEY, http_options=types.HttpOptions(timeout=10_000))
            names = [m.name.removeprefix("models/") for m in client.models.list()
                     if "generateContent" in (m.supported_actions or []) and "flash" in m.name
                     and not any(x in m.name for x in ("image", "tts", "audio", "live", "preview", "exp"))]
            _discovered = [n for n in sorted(names, key=lambda n: ("lite" in n, n), reverse=True)
                           if n != config.GEMINI_MODEL][:3]
        except Exception:  # noqa: BLE001
            _discovered = []
    return _discovered


# ---------------------------------------------------------------- Groq (OpenAI-compatible API)
GROQ_URL = "https://api.groq.com/openai/v1"
_groq_model: str | None = None


def _groq(system: str, user: str, timeout: float | None = None) -> str:
    """Groq runs open models (Llama etc.) on custom chips: typically 1-2 s per answer."""
    import httpx

    global _groq_model
    timeout = timeout or config.GROQ_TIMEOUT_S
    headers = {"Authorization": f"Bearer {config.GROQ_API_KEY}"}
    model = _groq_model or config.GROQ_MODEL
    body = {"model": model, "temperature": 0, "max_tokens": 1024,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    r = httpx.post(f"{GROQ_URL}/chat/completions", json=body, headers=headers, timeout=timeout)
    if r.status_code in (400, 404) and _MODEL_GONE.search(r.text) and _groq_model is None:
        # Model renamed/retired (NOT any error that merely mentions "model", e.g. a context-length error):
        # pick another large chat model this key can use, and remember it.
        ids = [m["id"] for m in httpx.get(f"{GROQ_URL}/models", headers=headers, timeout=min(timeout, 10))
               .json().get("data", [])]
        pref = [i for i in ids if any(k in i for k in ("llama-3.3", "gpt-oss-120b", "llama-4", "70b"))] or ids
        if not pref:
            raise RuntimeError(f"{r.status_code} {r.text[:300]} (and /models lists no replacement)")
        _groq_model = pref[0]
        body["model"] = _groq_model
        r = httpx.post(f"{GROQ_URL}/chat/completions", json=body, headers=headers, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code} {r.text[:300]}")
    choice = r.json()["choices"][0]
    if choice.get("finish_reason") == "length":
        raise EmptyAnswer(f"groq {body['model']}: answer cut off at max_tokens")
    content = choice["message"]["content"] or ""
    if not content.strip():
        raise EmptyAnswer(f"groq {body['model']}: empty answer")
    return content


# ---------------------------------------------------------------- circuit breaker
# If a provider/model just failed (overloaded, timeout), skip it for a few minutes instead of making
# every visitor wait through the same retries again.
_cooldown_until: dict[str, float] = {}
_last_probe: dict[str, float] = {}
COOLDOWN_S = 300
PROBE_INTERVAL_S = 10  # while EVERY provider is benched, send at most one test call this often


def _available(name: str) -> bool:
    return time.time() >= _cooldown_until.get(name, 0)


def _trip(name: str) -> None:
    _cooldown_until[name] = time.time() + COOLDOWN_S
    observability.log_event("llm_benched", provider=name, seconds=COOLDOWN_S)  # was silent; counted by metrics


def breaker_state() -> dict[str, dict]:
    """Read-only view of the circuit breaker for /api/health: every provider/model of LLM_PROVIDERS and whether
    it can be used right now. Never calls a provider (no quota used) and never starts the model discovery.
      {"groq": {"status": "cooling_down", "retry_in_s": 240}, "gemini:gemini-3.8-flash": {"status": "ok"}}
    status: ok | cooling_down (benched after a failure, for COOLDOWN_S) | no_key (listed, but no API key set)"""
    keys: dict[str, bool] = {}  # breaker key -> has an API key
    for provider in config.LLM_PROVIDERS:
        if provider == "groq":
            keys["groq"] = bool(config.GROQ_API_KEY)
        elif provider == "anthropic":
            keys["anthropic"] = bool(config.ANTHROPIC_API_KEY)
        elif provider == "gemini":
            models = [config.GEMINI_MODEL] + config.GEMINI_FALLBACK_MODELS + (_discovered or [])
            for m in dict.fromkeys(models):
                keys[f"gemini:{m}"] = bool(config.GEMINI_API_KEY)
    now = time.time()
    state = {}
    for key, has_key in keys.items():
        left = _cooldown_until.get(key, 0) - now
        if not has_key:
            state[key] = {"status": "no_key"}
        elif left > 0:
            state[key] = {"status": "cooling_down", "retry_in_s": int(left) + 1}
        else:
            state[key] = {"status": "ok"}
    return state


def _candidates():
    """(breaker key, model label, call(system, user, timeout)) in order of preference. A generator, so
    Gemini backup models are only looked up when the configured ones were skipped or failed."""
    for provider in config.LLM_PROVIDERS:
        if provider == "groq" and config.GROQ_API_KEY:
            yield "groq", lambda: f"groq/{_groq_model or config.GROQ_MODEL}", _groq
        elif provider == "anthropic" and config.ANTHROPIC_API_KEY:
            yield "anthropic", lambda: config.ANTHROPIC_MODEL, _anthropic
        elif provider == "gemini" and config.GEMINI_API_KEY:
            models = [config.GEMINI_MODEL] + [m for m in config.GEMINI_FALLBACK_MODELS if m != config.GEMINI_MODEL]
            for m in models:
                yield f"gemini:{m}", (lambda m=m: m), (lambda s, u, timeout, m=m: _gemini(m, s, u, timeout=timeout))
            for m in _discover_backup_models():
                if m not in models:
                    yield f"gemini:{m}", (lambda m=m: m), (lambda s, u, timeout, m=m: _gemini(m, s, u, timeout=timeout))


def _provider_timeout(key: str) -> float:
    return config.GROQ_TIMEOUT_S if key == "groq" else config.LLM_TIMEOUT_S


def generate(system: str, user: str) -> str:
    """Try providers in the configured order; a provider that fails is skipped for a few minutes.
    Returns a non-empty answer or raises LLMUnavailable."""
    deadline = _deadline()
    errors: list[str] = []
    benched: list[tuple] = []

    def attempt(key, label, call, attempts: int) -> str:
        def once():
            left = deadline - time.time()
            if left < MIN_ATTEMPT_S:
                raise LLMUnavailable(f"{key}: no time left in the {config.LLM_DEADLINE_S:.0f} s budget")
            return call(system, user, timeout=min(_provider_timeout(key), left))
        out = strip_reasoning(_with_retries(once, attempts=attempts, deadline=deadline))
        if not out:
            raise EmptyAnswer(f"{key}: the answer was only reasoning notes")
        _cooldown_until.pop(key, None)  # it works (again)
        _record_model(label())
        return out

    for key, label, call in _candidates():
        if not _available(key):
            benched.append((key, label, call))
            continue
        if deadline - time.time() < MIN_ATTEMPT_S:
            errors.append(f"{key}: not tried, time budget used up")
            break
        try:
            return attempt(key, label, call, attempts=2)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{key}: {type(e).__name__}: {str(e)[:200]}")
            observability.log_event("llm_failed", provider=key, error=errors[-1])
            if not _is_request_error(e) and not isinstance(e, LLMUnavailable):
                _trip(key)

    if benched and not errors:
        # Everything is benched after a recent outage: still test ONE provider now and then, so the site
        # recovers as soon as a provider does, instead of failing every question for COOLDOWN_S.
        for key, label, call in benched:
            if time.time() - _last_probe.get(key, float("-inf")) >= PROBE_INTERVAL_S:
                _last_probe[key] = time.time()
                try:
                    return attempt(key, label, call, attempts=1)
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{key} (probe): {type(e).__name__}: {str(e)[:200]}")
                    if not _is_request_error(e):
                        _trip(key)
                break
        else:
            errors.append("all providers are cooling down after recent failures")

    if not errors and not benched:
        raise LLMUnavailable("no LLM provider configured (check LLM_PROVIDER and the API keys)")
    raise LLMUnavailable("all LLM providers failed: " + "; ".join(errors), retry_after_s=PROBE_INTERVAL_S)
