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


def generate(system: str, user: str) -> str:
    global last_model_used
    if config.LLM_PROVIDER == "anthropic":
        last_model_used = config.ANTHROPIC_MODEL
        return _with_retries(lambda: _anthropic(system, user))

    # Gemini: main model with retries -> configured fallbacks -> auto-discovered backups.
    tried: list[str] = []
    candidates = [config.GEMINI_MODEL] + config.GEMINI_FALLBACK_MODELS
    error: Exception | None = None
    i = 0
    while i < len(candidates):
        model = candidates[i]
        tried.append(model)
        try:
            out = _with_retries(lambda: _gemini(model, system, user), attempts=3 if i == 0 else 2)
            last_model_used = model
            return out
        except Exception as e:  # noqa: BLE001
            error = e
            if i == len(candidates) - 1 and len(tried) == len(candidates):
                candidates += [m for m in _discover_backup_models() if m not in candidates]
        i += 1
    raise error  # type: ignore[misc]
