"""Group 6: the LLM wrapper (app/llm.py) -- retries, Groq -> Gemini fallback, circuit breaker, and what
happens when a provider times out, rate-limits, returns garbage or is down together with the backup.

Nothing here touches the network: Groq is a fake `httpx.post/get`, Gemini is a fake `_gemini` (or a fake
`google.genai.Client` for the tests of `_gemini` itself), and time is a fake clock, so a "30 s timeout"
costs nothing and waits can be measured exactly.

Tests marked `xfail(strict=True)` describe a KNOWN BUG (red first, see TESTING.md). When the fix lands
they start passing, pytest reports XPASS as a failure, and the marker must be removed.
"""
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors

from app import config, llm

_real_gemini = llm._gemini  # captured at import, before the env fixture swaps in FakeGemini
BUG = lambda reason: pytest.mark.xfail(strict=True, reason=reason)  # noqa: E731


# ------------------------------------------------------------------ fakes
class FakeClock:
    """Replaces the `time` module inside llm.py: sleep() and timeouts only move the clock forward."""
    def __init__(self):
        self.now = 1_000_000.0
        self.slept: list[float] = []

    def time(self):
        return self.now

    monotonic = perf_counter = time

    def sleep(self, s):
        self.slept.append(s)
        self.now += s


def _script_step(steps: list):
    """Next scripted step; the last one repeats forever."""
    return steps.pop(0) if len(steps) > 1 else steps[0]


class FakeGroq:
    """Stands in for httpx.post / httpx.get. `script` = list of httpx.Response or exceptions.
    A raised httpx.TimeoutException advances the clock by the timeout the caller passed."""
    def __init__(self, clock):
        self.clock = clock
        self.script: list = [ok("groq answer [1]")]
        self.models: list[str] = ["llama-3.3-70b-versatile"]
        self.posts: list[dict] = []
        self.gets: list[str] = []

    def post(self, url, json=None, headers=None, timeout=None, **_):
        self.posts.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        step = _script_step(self.script)
        if isinstance(step, httpx.TimeoutException):
            self.clock.now += timeout if isinstance(timeout, (int, float)) else config.LLM_TIMEOUT_S
        if isinstance(step, Exception):
            raise step
        return step

    def get(self, url, headers=None, timeout=None, **_):
        self.gets.append(url)
        return httpx.Response(200, json={"data": [{"id": m} for m in self.models]})


class FakeGemini:
    """Stands in for llm._gemini(model, system, user). `script[model]` = list of str or exceptions
    ("*" = any model not listed). Timeouts advance the clock like FakeGroq."""
    def __init__(self, clock):
        self.clock = clock
        self.script: dict[str, list] = {"*": ["gemini answer [1]"]}
        self.calls: list[str] = []

    def __call__(self, model, system, user, **kw):
        self.calls.append(model)
        step = _script_step(self.script.get(model, self.script["*"]))
        if isinstance(step, httpx.TimeoutException):
            self.clock.now += kw.get("timeout") or config.LLM_TIMEOUT_S
        if isinstance(step, Exception):
            raise step
        return step


def ok(text, finish="stop"):
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": text},
                                                  "finish_reason": finish}]})


def err(code, text):
    return httpx.Response(code, text=text)


def timeout():
    return httpx.ReadTimeout("The read operation timed out")


def gemini_503():
    return genai_errors.ServerError(503, {"error": {"code": 503, "message": "The model is overloaded.",
                                                    "status": "UNAVAILABLE"}})


def gemini_429():
    return genai_errors.ClientError(429, {"error": {"code": 429, "message": "Quota exceeded",
                                                    "status": "RESOURCE_EXHAUSTED"}})


class NoNetworkClient:
    def __init__(self, *a, **k):
        raise AssertionError("a test tried to create a real google.genai.Client")


@pytest.fixture
def env(monkeypatch):
    """Production-like config (groq first, gemini second), clean module state, fake clock and providers."""
    clock = FakeClock()
    groq, gemini = FakeGroq(clock), FakeGemini(clock)
    monkeypatch.setattr(llm, "time", clock)
    monkeypatch.setattr(httpx, "post", groq.post)
    monkeypatch.setattr(httpx, "get", groq.get)
    monkeypatch.setattr(llm, "_gemini", gemini)
    monkeypatch.setattr("google.genai.Client", NoNetworkClient)
    for name, value in {"_cooldown_until": {}, "_last_probe": {}, "_groq_model": None, "_discovered": [],
                        "_thinking_supported": {}, "last_model_used": ""}.items():
        monkeypatch.setattr(llm, name, value)
    for name, value in {"LLM_PROVIDERS": ["groq", "gemini"], "GROQ_API_KEY": "test-groq-key",
                        "GROQ_MODEL": "llama-3.3-70b-versatile", "GEMINI_API_KEY": "test-gemini-key",
                        "GEMINI_MODEL": "gemini-main", "GEMINI_FALLBACK_MODELS": [], "LLM_TIMEOUT_S": 30.0,
                        "GROQ_TIMEOUT_S": 12.0, "LLM_DEADLINE_S": 30.0,  # agreed limits, not the local .env
                        "GEMINI_THINKING": "LOW"}.items():
        monkeypatch.setattr(config, name, value)
    return SimpleNamespace(clock=clock, groq=groq, gemini=gemini)


# ================================================================== 1. retries (_with_retries)
class Flaky:
    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    def __call__(self):
        self.calls += 1
        step = _script_step(self.outcomes)
        if isinstance(step, Exception):
            raise step
        return step


@pytest.mark.parametrize("error", [
    RuntimeError("429 Rate limit reached for model"), RuntimeError("500 Internal Server Error"),
    RuntimeError("502 Bad Gateway"), RuntimeError("503 Service Unavailable"), RuntimeError("504 Gateway Timeout"),
    RuntimeError("Anthropic: overloaded_error"), gemini_503(), gemini_429(),
], ids=["429", "500", "502", "503", "504", "overloaded", "gemini-503", "gemini-429"])
def test_transient_errors_are_retried_after_a_backoff(env, error):
    """Catches: a rate limit / overload (a normal event on free tiers) failing the request at once
    instead of waiting 1 s and trying again."""
    fn = Flaky(error, "answer")
    assert llm._with_retries(fn, attempts=2) == "answer"
    assert fn.calls == 2 and env.clock.slept == [1]


def test_backoff_doubles_and_the_last_error_is_raised(env):
    """Catches: a broken backoff (no wait, or waits that don't grow) and a swallowed final error."""
    fn = Flaky(RuntimeError("503 overloaded"))
    with pytest.raises(RuntimeError, match="503"):
        llm._with_retries(fn, attempts=4)
    assert fn.calls == 4 and env.clock.slept == [1, 2, 4]


@pytest.mark.parametrize("error", [RuntimeError("401 Invalid API Key"), RuntimeError("403 Forbidden"),
                                   RuntimeError("400 Bad request: messages is required")])
def test_permanent_errors_are_not_retried(env, error):
    """Catches: retrying a wrong key or a bad request, which only adds waiting before the same failure."""
    fn = Flaky(error, "never reached")
    with pytest.raises(RuntimeError):
        llm._with_retries(fn, attempts=4)
    assert fn.calls == 1 and env.clock.slept == []


def test_a_timeout_is_not_repeated_on_the_same_provider(env):
    """Catches: a 30 s timeout being retried on the same provider (60 s+ of waiting) instead of moving
    on to the backup provider."""
    fn = Flaky(timeout(), "never reached")
    with pytest.raises(httpx.TimeoutException):
        llm._with_retries(fn, attempts=2)
    assert fn.calls == 1


@pytest.mark.parametrize("message", ["400 Request too large: 15004 tokens requested, limit 6000",
                                     "Request too large: 5030 tokens requested"])  # no leading code
def test_a_status_code_is_not_found_inside_another_number(env, message):
    """Catches: a permanent 400 ("request too large: 15004 tokens") being retried because the text
    happens to contain "500" (or "5030" contains "503")."""
    fn = Flaky(RuntimeError(message), "never reached")
    with pytest.raises(RuntimeError):
        llm._with_retries(fn, attempts=2)
    assert fn.calls == 1


# ================================================================== 2. Groq provider (_groq)
def test_groq_request_is_deterministic_authenticated_and_time_limited(env):
    """Catches: a lost timeout (a hung request blocks a worker forever), temperature != 0 (answers
    change between identical questions) or a missing key header."""
    assert llm._groq("SYSTEM", "USER") == "groq answer [1]"
    sent = env.groq.posts[0]
    assert sent["url"].endswith("/chat/completions")
    assert isinstance(sent["timeout"], (int, float)) and sent["timeout"] > 0
    assert sent["headers"]["Authorization"] == "Bearer test-groq-key"
    assert sent["json"]["temperature"] == 0 and sent["json"]["model"] == "llama-3.3-70b-versatile"
    assert sent["json"]["messages"] == [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "USER"}]


def test_hanging_groq_costs_at_most_15_s_before_gemini_answers(env):
    """Catches: a hanging Groq eating the whole time budget, so Gemini has no time left to answer."""
    env.groq.script = [timeout()]
    start = env.clock.now
    assert llm.generate("s", "u") == "gemini answer [1]"
    assert env.clock.now - start <= 15


def test_groq_http_error_carries_the_status_code(env):
    """Catches: an error message without the status code, which would make _with_retries treat a 503
    as permanent (no retry) and the UI show "something went wrong" instead of "busy, retry"."""
    env.groq.script = [err(503, "Service Unavailable")]
    with pytest.raises(RuntimeError, match=r"^503"):
        llm._groq("s", "u")


def test_retired_groq_model_is_replaced_and_remembered(env):
    """Catches: Groq retiring a model (it happens every few months) taking the primary provider down."""
    env.groq.script = [err(400, '{"error":{"message":"The model `llama-3.3-70b-versatile` has been '
                                'decommissioned","code":"model_decommissioned"}}'), ok("new model answer [1]")]
    env.groq.models = ["whisper-large-v3", "openai/gpt-oss-120b", "llama-3.1-8b-instant"]
    assert llm.generate("s", "u") == "new model answer [1]"
    assert llm.last_model_used == "groq/openai/gpt-oss-120b"
    env.groq.script = [ok("second answer [1]")]
    assert llm.generate("s", "u") == "second answer [1]"
    assert len(env.groq.gets) == 1 and env.groq.posts[-1]["json"]["model"] == "openai/gpt-oss-120b"


def test_context_length_error_does_not_swap_the_production_model(env):
    """Catches: one oversized request ("This model's maximum context length is ...") permanently
    switching every later request to whatever model /models lists first."""
    env.groq.script = [err(400, "This model's maximum context length is 8192 tokens. "
                                "Please reduce the length of the messages.")]
    env.groq.models = ["deepseek-r1-distill-llama-70b"]
    llm.generate("s", "u")
    assert env.groq.gets == [] and llm._groq_model is None


def test_malformed_groq_success_body_falls_back_to_gemini(env):
    """Catches: a 200 with an unexpected body (proxy HTML, error JSON) crashing the request instead of
    being treated as a provider failure."""
    env.groq.script = [httpx.Response(200, json={"error": {"message": "upstream hiccup"}})]
    assert llm.generate("s", "u") == "gemini answer [1]"


@pytest.mark.parametrize("content", ["", None, "   "])
def test_empty_groq_answer_falls_through_to_gemini(env, content):
    """Catches: a provider glitch being shown to the user as "not found in the leaflets"."""
    env.groq.script = [ok(content, finish="length")]
    assert llm.generate("s", "u") == "gemini answer [1]"


def test_truncated_groq_answer_is_not_returned_as_complete(env):
    """Catches: a cut-off medical answer ("Do not take more than 3 tablets in") reaching the user."""
    env.groq.script = [ok("Adults: 1 tablet [1]. Do not take more than 3 tablets in", finish="length")]
    assert llm.generate("s", "u") == "gemini answer [1]"


# ================================================================== 3. Gemini provider (_gemini, _try_gemini)
class FakeGenai:
    """A fake google.genai.Client. `answers[model]` = list of response texts (None = blocked/empty) or
    exceptions. Records each client's http_options and each call's config."""
    def __init__(self):
        self.answers: dict[str, list] = {}
        self.listed: list = []
        self.clients: list = []
        self.calls: list = []
        self.list_calls = 0
        fake = self

        class Models:
            def generate_content(self, model, contents, config):
                fake.calls.append({"model": model, "config": config})
                step = _script_step(fake.answers[model])
                if isinstance(step, Exception):
                    raise step
                return SimpleNamespace(text=step)

            def list(self):
                fake.list_calls += 1
                return fake.listed

        class Client:
            def __init__(self, api_key=None, http_options=None, **_):
                fake.clients.append(http_options)
                self.models = Models()
        self.Client = Client


@pytest.fixture
def genai_fake(env, monkeypatch):
    """The REAL _gemini / _try_gemini code running against a fake SDK client."""
    fake = FakeGenai()
    monkeypatch.setattr("google.genai.Client", fake.Client)
    monkeypatch.setattr(llm, "_gemini", _real_gemini)
    return fake


def test_gemini_request_is_deterministic_and_time_limited(genai_fake):
    """Catches: a Gemini call without a timeout (the SDK default waits minutes) or with sampling on."""
    genai_fake.answers["gemini-main"] = ["answer [1]"]
    assert llm._gemini("gemini-main", "SYSTEM", "USER") == "answer [1]"
    assert genai_fake.clients[0].timeout == 30_000  # milliseconds
    cfg = genai_fake.calls[0]["config"]
    assert cfg.temperature == 0 and cfg.system_instruction == "SYSTEM" and cfg.thinking_config is not None


def test_model_without_thinking_support_is_retried_without_it_and_remembered(genai_fake):
    """Catches: a lite/older backup model rejecting the thinking level, so the backup never works."""
    genai_fake.answers["gemini-lite"] = [RuntimeError("400 INVALID_ARGUMENT: thinking_level is not supported"),
                                         "answer [1]", "again [1]"]
    assert llm._gemini("gemini-lite", "s", "u") == "answer [1]"
    assert llm._gemini("gemini-lite", "s", "u") == "again [1]"
    assert [c["config"].thinking_config is None for c in genai_fake.calls] == [False, True, True]


def test_failing_main_gemini_model_falls_back_to_the_configured_backup(env, monkeypatch):
    """Catches: GEMINI_FALLBACK_MODELS being ignored, and the UI naming the wrong model."""
    monkeypatch.setattr(config, "LLM_PROVIDERS", ["gemini"])
    monkeypatch.setattr(config, "GEMINI_FALLBACK_MODELS", ["gemini-backup"])
    env.gemini.script["gemini-main"] = [gemini_503()]
    env.gemini.script["gemini-backup"] = ["backup answer [1]"]
    assert llm.generate("s", "u") == "backup answer [1]"
    assert env.gemini.calls == ["gemini-main", "gemini-main", "gemini-backup"]  # 503 is retried once
    assert llm.last_model_used == "gemini-backup"


def test_discovered_backups_are_tried_after_the_configured_ones_and_listed_once(genai_fake, monkeypatch):
    """Catches: model discovery running on every request (an extra API call while Gemini is already
    struggling) or offering image/tts/preview models that cannot answer."""
    monkeypatch.setattr(llm, "_discovered", None)
    monkeypatch.setattr(config, "LLM_PROVIDERS", ["gemini"])
    genai_fake.listed = [SimpleNamespace(name=f"models/{n}", supported_actions=["generateContent"]) for n in
                         ("gemini-main", "gemini-2.5-flash", "gemini-2.5-flash-image", "gemini-3-flash-preview")]
    genai_fake.answers = {"gemini-main": [gemini_503()], "gemini-2.5-flash": ["discovered answer [1]"]}
    assert llm.generate("s", "u") == "discovered answer [1]"
    assert llm._discovered == ["gemini-2.5-flash"] and genai_fake.list_calls == 1
    llm._cooldown_until.clear()
    llm.generate("s", "u")
    assert genai_fake.list_calls == 1


def test_model_discovery_is_time_limited(genai_fake, monkeypatch):
    """Catches: the backup-model lookup hanging a request for minutes during a Gemini outage."""
    monkeypatch.setattr(llm, "_discovered", None)
    llm._discover_backup_models()
    assert genai_fake.clients and genai_fake.clients[0] is not None and genai_fake.clients[0].timeout


def test_blocked_or_empty_gemini_answer_moves_to_the_next_model(genai_fake, monkeypatch):
    """Catches: a safety-blocked reply to an overdose question becoming a silent "not found"."""
    monkeypatch.setattr(config, "LLM_PROVIDERS", ["gemini"])
    monkeypatch.setattr(config, "GEMINI_FALLBACK_MODELS", ["gemini-backup"])
    genai_fake.answers = {"gemini-main": [None], "gemini-backup": ["backup answer [1]"]}
    assert llm.generate("s", "u") == "backup answer [1]"



# ================================================================== 4. fallback chain + circuit breaker (generate)
def test_groq_answer_wins_and_gemini_is_not_called(env):
    """Catches: a wrong provider order (slow Gemini first) or both providers being called."""
    assert llm.generate("s", "u") == "groq answer [1]"
    assert env.gemini.calls == [] and llm.last_model_used == "groq/llama-3.3-70b-versatile"


def test_groq_timeout_falls_back_to_gemini_and_benches_groq_for_a_while(env):
    """Catches: every visitor waiting 30 s for a hanging Groq before getting the Gemini answer."""
    env.groq.script = [timeout()]
    assert llm.generate("s", "u") == "gemini answer [1]"
    assert len(env.groq.posts) == 1 and llm.last_model_used == "gemini-main"

    env.clock.now += 60
    assert llm.generate("s", "u") == "gemini answer [1]"
    assert len(env.groq.posts) == 1, "groq is cooling down, must not be called"

    env.clock.now += llm.COOLDOWN_S
    env.groq.script = [ok("groq is back [1]")]
    assert llm.generate("s", "u") == "groq is back [1]"


def test_groq_429_is_retried_once_then_gemini_answers(env):
    """Catches: a rate-limited Groq failing the request although Gemini is available."""
    env.groq.script = [err(429, '{"error":{"message":"Rate limit reached","type":"tokens"}}')]
    assert llm.generate("s", "u") == "gemini answer [1]"
    assert len(env.groq.posts) == 2 and env.clock.slept == [1]


def test_missing_groq_key_skips_groq_silently(env, monkeypatch):
    """Catches: an unconfigured provider costing a failed request (and a log line) on every question."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    assert llm.generate("s", "u") == "gemini answer [1]" and env.groq.posts == []


def test_total_outage_raises_instead_of_returning_an_empty_answer(env):
    """Catches: an outage being turned into "" -- rag.ask would then tell the user the leaflets don't
    cover the question, which is false."""
    env.groq.script = [err(503, "Service Unavailable")]
    env.gemini.script["*"] = [gemini_503()]
    with pytest.raises(Exception):
        llm.generate("s", "u")


def test_total_outage_raises_one_typed_error_naming_every_provider(env):
    """Catches: the user seeing only the LAST provider's raw error, and the API being unable to map an
    outage to 503 + Retry-After."""
    env.groq.script = [err(429, "Rate limit reached")]
    env.gemini.script["*"] = [gemini_503()]
    with pytest.raises(llm.LLMUnavailable) as exc:
        llm.generate("s", "u")
    assert "groq" in str(exc.value) and "gemini" in str(exc.value)
    assert exc.value.retry_after_s > 0


def test_total_outage_is_bounded_by_one_deadline(env, monkeypatch):
    """Catches: the user staring at the spinner for minutes while every model times out in turn."""
    monkeypatch.setattr(llm, "_discovered", ["gemini-b1", "gemini-b2"])
    env.groq.script = [timeout()]
    env.gemini.script["*"] = [timeout()]
    start = env.clock.now
    with pytest.raises(Exception):
        llm.generate("s", "u")
    assert env.clock.now - start <= 30 + 1  # +1 s for a backoff sleep


def test_after_a_total_outage_the_next_request_still_probes_a_provider(env):
    """Catches: a 10-second blip of both providers locking the whole site for COOLDOWN_S (300 s)."""
    env.groq.script = [err(503, "Service Unavailable")]
    env.gemini.script["*"] = [gemini_503()]
    with pytest.raises(Exception):
        llm.generate("s", "u")
    env.clock.now += 10
    env.groq.script = [ok("groq recovered [1]")]
    env.gemini.script["*"] = ["gemini recovered [1]"]
    assert llm.generate("s", "u") in ("groq recovered [1]", "gemini recovered [1]")


def test_a_bad_request_does_not_bench_groq_for_everyone(env):
    """Catches: one unusual question moving all traffic to the slower backup."""
    env.groq.script = [err(400, "Please reduce the length of the messages or completion.")]
    assert llm.generate("s", "u") == "gemini answer [1]"
    assert llm._available("groq")
