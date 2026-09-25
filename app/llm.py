"""Thin wrapper around the LLM API. Swapping providers = changing LLM_PROVIDER in .env."""
import time

from . import config

RETRYABLE = ("429", "500", "502", "503", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded")


def _with_retries(fn, attempts: int = 4):
    """Free-tier APIs are sometimes overloaded (HTTP 503) or rate-limited (429).
    Retry with exponential backoff: wait 1s, 2s, 4s between attempts."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == attempts - 1 or not any(code in str(e) for code in RETRYABLE):
                raise
            time.sleep(2 ** i)


_thinking_supported: dict[str, bool] = {}


def _gemini(model: str, system: str, user: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.GEMINI_API_KEY,
                          http_options=types.HttpOptions(timeout=int(config.LLM_TIMEOUT_S * 1000)))
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
    return resp.text or ""


def _anthropic(system: str, user: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY, timeout=config.LLM_TIMEOUT_S)
    msg = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=800,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


_discovered: list[str] | None = None
last_model_used: str = ""


def _discover_backup_models() -> list[str]:
    """If no GEMINI_FALLBACK_MODELS are configured, ask the API which other 'flash' models this key
    can use (lighter/older ones are less often overloaded). Cached after the first call."""
    global _discovered
    if _discovered is None:
        try:
            from google import genai

            client = genai.Client(api_key=config.GEMINI_API_KEY)
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


def _groq(system: str, user: str) -> str:
    """Groq runs open models (Llama etc.) on custom chips: typically 1-2 s per answer."""
    import httpx

    global _groq_model
    headers = {"Authorization": f"Bearer {config.GROQ_API_KEY}"}
    model = _groq_model or config.GROQ_MODEL
    body = {"model": model, "temperature": 0, "max_tokens": 1024,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    r = httpx.post(f"{GROQ_URL}/chat/completions", json=body, headers=headers, timeout=config.LLM_TIMEOUT_S)
    if r.status_code in (400, 404) and "model" in r.text.lower() and _groq_model is None:
        # Model renamed/retired: pick another large chat model this key can use, and remember it.
        ids = [m["id"] for m in httpx.get(f"{GROQ_URL}/models", headers=headers, timeout=15).json().get("data", [])]
        pref = [i for i in ids if any(k in i for k in ("llama-3.3", "gpt-oss-120b", "llama-4", "70b"))] or ids
        _groq_model = pref[0]
        body["model"] = _groq_model
        r = httpx.post(f"{GROQ_URL}/chat/completions", json=body, headers=headers, timeout=config.LLM_TIMEOUT_S)
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code} {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"] or ""


# ---------------------------------------------------------------- circuit breaker
# If a provider/model just failed (overloaded, timeout), skip it for a few minutes instead of making
# every visitor wait through the same retries again.
_cooldown_until: dict[str, float] = {}
COOLDOWN_S = 300


def _available(name: str) -> bool:
    return time.time() >= _cooldown_until.get(name, 0)


def _trip(name: str) -> None:
    _cooldown_until[name] = time.time() + COOLDOWN_S


def _try_gemini(system: str, user: str) -> str:
    global last_model_used
    candidates = [config.GEMINI_MODEL] + config.GEMINI_FALLBACK_MODELS
    candidates += [m for m in _discover_backup_models() if m not in candidates] if not _available(
        f"gemini:{config.GEMINI_MODEL}") else []
    error: Exception | None = None
    for i, model in enumerate(candidates):
        key = f"gemini:{model}"
        if not _available(key):
            continue
        try:
            out = _with_retries(lambda: _gemini(model, system, user), attempts=2)
            last_model_used = model
            return out
        except Exception as e:  # noqa: BLE001
            error, _ = e, _trip(key)
            if i == len(candidates) - 1:
                candidates += [m for m in _discover_backup_models() if m not in candidates]
    raise error or RuntimeError("all Gemini models are cooling down after recent failures")


def generate(system: str, user: str) -> str:
    """Try providers in the configured order; a provider that fails is skipped for a few minutes."""
    global last_model_used
    error: Exception | None = None
    for provider in config.LLM_PROVIDERS:
        try:
            if provider == "groq":
                if not config.GROQ_API_KEY or not _available("groq"):
                    continue
                out = _with_retries(lambda: _groq(system, user), attempts=2)
                last_model_used = f"groq/{_groq_model or config.GROQ_MODEL}"
                return out
            if provider == "anthropic":
                out = _with_retries(lambda: _anthropic(system, user), attempts=2)
                last_model_used = config.ANTHROPIC_MODEL
                return out
            if provider == "gemini":
                return _try_gemini(system, user)
        except Exception as e:  # noqa: BLE001
            error = e
            if provider == "groq":
                _trip("groq")
    raise error or RuntimeError("no LLM provider configured")
