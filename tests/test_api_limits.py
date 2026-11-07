"""Point 4 of the API hardening (group 7d): a limit on questions per visitor (429 "too many, wait").

Why: every question costs up to 4-5 LLM calls on the free Groq/Gemini keys; one script could use up the quota
and every real user would get "busy". Limits: ASK_LIMIT_PER_MINUTE (10) and ASK_LIMIT_PER_DAY (200), 0 = off.

Who is "one visitor": the LAST address in the X-Forwarded-For note (Render's proxy adds the real one at the END;
a bot can only add fake ones at the START). Without the note: the direct caller. IPv6 is grouped by /64 (one
home connection gets a whole block of addresses). NOT yet checked on the live Render server - see TESTING.md.

No network: rag.* and the mail service are fakes; the clock is fake (a "day" costs nothing).
"""
import re
import threading

import pytest
from fastapi.testclient import TestClient

from app import config, feedback, limits, main, rag, spam

OK_Q = "What is metformin?"


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t

    def wait(self, seconds):
        self.t += seconds


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(limits, "now", c)
    return c


@pytest.fixture(autouse=True)
def fresh(monkeypatch, clock):
    """Default limits, empty counters, fake pipeline + mail for every test. `asked` counts questions that
    reached the pipeline."""
    monkeypatch.setattr(config, "ASK_LIMIT_PER_MINUTE", 10)
    monkeypatch.setattr(config, "ASK_LIMIT_PER_DAY", 200)
    limits.ask_limiter.reset()
    spam.reset()
    asked = []
    monkeypatch.setattr(rag, "ask", lambda question, *a, **k: asked.append(question) or {"answer": "ok [1]"})
    monkeypatch.setattr(rag, "library", lambda market="us": [])
    monkeypatch.setattr(rag, "random_fact", lambda seen=None, lang="en": {})
    monkeypatch.setattr(config, "RESEND_API_KEY", "key")
    monkeypatch.setattr(config, "FEEDBACK_TO", "owner@example.com")

    class R:
        status_code = 200
    monkeypatch.setattr(feedback.httpx, "post", lambda url, json, timeout, headers: R())
    yield asked
    limits.ask_limiter.reset()
    spam.reset()


@pytest.fixture
def asked(fresh):
    return fresh


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


def ask(client, ip=None, xff=None):
    headers = {"X-Forwarded-For": xff or ip} if (xff or ip) else {}
    return client.post("/api/ask", json={"question": OK_Q}, headers=headers)


# ================================================================== 1. who is "one visitor"

class FakeRequest:
    def __init__(self, xff=None, host="10.0.0.5"):
        from starlette.datastructures import Headers
        raw = [("x-forwarded-for", v) for v in ([xff] if isinstance(xff, str) else (xff or []))]
        self.headers = Headers(raw=[(k.encode(), v.encode()) for k, v in raw])
        self.client = type("C", (), {"host": host})()


@pytest.mark.parametrize("xff, expected", [
    ("203.0.113.7", "203.0.113.7"),                          # Render's note: the visitor
    ("1.1.1.1, 203.0.113.7", "203.0.113.7"),                 # a bot wrote "1.1.1.1" itself, Render appended the real one
    ("1.1.1.1,2.2.2.2,  203.0.113.7 ", "203.0.113.7"),       # several fakes, odd spaces
    (["1.1.1.1", "5.5.5.5, 203.0.113.7"], "203.0.113.7"),    # the note sent twice: the last entry of the last one
    ("2001:db8::1", "2001:db8::1"),
    (None, "10.0.0.5"),                                      # no note (local run): the direct caller
    ("not-an-ip", "10.0.0.5"),                               # garbage: the direct caller, not the garbage
    ("203.0.113.7, ", "10.0.0.5"),                           # empty last entry: same
])
def test_the_visitor_is_the_last_address_in_the_proxy_note(xff, expected):
    """Catches: using the FIRST address (a bot can write anything there) or trusting garbage."""
    assert limits.client_ip(FakeRequest(xff)) == expected


def test_ipv6_addresses_of_one_connection_count_as_one_visitor():
    """Catches: a bot on IPv6 changing the last part of its address (it owns billions) to get a new counter."""
    assert limits.visitor_key("2001:db8:1:2::1") == limits.visitor_key("2001:db8:1:2:ffff::99")
    assert limits.visitor_key("2001:db8:1:2::1") != limits.visitor_key("2001:db8:1:3::1")
    assert limits.visitor_key("203.0.113.7") != limits.visitor_key("203.0.113.8")


@pytest.mark.parametrize("ip, masked", [("203.0.113.7", "203.0.113.x"), ("2001:db8:1:2::1", "2001:db8:1:x")])
def test_the_log_never_shows_a_full_address(ip, masked):
    """Catches: full visitor addresses (personal data) written to the log."""
    assert limits.mask(ip) == masked


# ================================================================== 2. the limits on /api/ask

def test_the_11th_question_in_a_minute_is_a_429_the_ui_shows_as_busy(client, asked):
    """Catches: no limit (a script can use up the LLM quota), or a 429 the UI doesn't show as "busy, retry"."""
    for _ in range(10):
        assert ask(client, "203.0.113.7").status_code == 200
    r = ask(client, "203.0.113.7")
    assert r.status_code == 429
    body = r.json()
    assert body["error"] == "RATE_LIMITED" and "429" in body["detail"]
    assert 0 < int(r.headers["Retry-After"]) <= 60
    js = (config.ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    busy = re.search(r"const busy = /(.+?)/i\.test", js).group(1)  # the UI's own "busy" pattern
    assert re.search(busy, body["detail"], re.I), "the UI must show 'busy, try again' + its retry button"
    assert len(asked) == 10  # the 11th never reached the LLM


def test_after_a_minute_the_visitor_can_ask_again(client, clock):
    """Catches: a minute limit that never resets."""
    for _ in range(10):
        ask(client, "203.0.113.7")
    assert ask(client, "203.0.113.7").status_code == 429
    clock.wait(61)
    assert ask(client, "203.0.113.7").status_code == 200


def test_the_wait_time_is_exact(client, clock):
    """Catches: a wrong Retry-After (the UI would tell the user to wait too long or too little)."""
    for _ in range(10):
        ask(client, "203.0.113.7")
        clock.wait(2)                      # 10 questions over 20 s; the first was 20 s ago
    r = ask(client, "203.0.113.7")
    assert r.status_code == 429 and int(r.headers["Retry-After"]) == 40


def test_the_201st_question_in_a_day_is_a_429_with_hours_to_wait(client, clock, asked):
    """Catches: no daily limit (a slow script: 10 a minute = 14,400 a day)."""
    for i in range(20):                    # the first 20 through the real API...
        assert ask(client, "203.0.113.7").status_code == 200, i
        clock.wait(7)                      # below 10 a minute
    day_rules = [(config.ASK_LIMIT_PER_MINUTE, 60), (config.ASK_LIMIT_PER_DAY, 24 * 3600)]
    for _ in range(180):                   # ...the next 180 straight into the counter (fast), same rules
        assert limits.ask_limiter.hit(limits.visitor_key("203.0.113.7"), day_rules) == 0
        clock.wait(7)
    r = ask(client, "203.0.113.7")
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) > 3600 and " h" in r.json()["detail"]
    clock.wait(24 * 3600)
    assert ask(client, "203.0.113.7").status_code == 200
    assert len(asked) == 21


def test_visitors_have_separate_counters(client):
    """Catches: one shared counter (one bot would block every real user)."""
    for _ in range(10):
        ask(client, "203.0.113.7")
    assert ask(client, "203.0.113.7").status_code == 429
    assert ask(client, "198.51.100.20").status_code == 200


def test_fake_addresses_at_the_start_of_the_note_do_not_help_a_bot(client):
    """Catches: a bot getting a fresh counter by writing a new fake address into the note each time."""
    for i in range(10):
        assert ask(client, xff=f"10.9.9.{i}, 203.0.113.7").status_code == 200
    assert ask(client, xff="10.9.9.99, 203.0.113.7").status_code == 429


def test_blocked_attempts_do_not_extend_the_wait(client, clock):
    """Catches: counting refused requests - a user pressing "retry" would stay blocked forever."""
    for _ in range(10):
        ask(client, "203.0.113.7")
    for _ in range(19):                    # "retry" every 3 s during the whole minute
        clock.wait(3)
        assert ask(client, "203.0.113.7").status_code == 429
    clock.wait(4)                          # 61 s after the 10 questions
    assert ask(client, "203.0.113.7").status_code == 200


def test_invalid_requests_do_not_use_up_the_limit(client):
    """Catches: 400 requests (typos, empty questions) eating a visitor's questions."""
    for _ in range(15):
        assert client.post("/api/ask", json={"question": "  "}, headers={"X-Forwarded-For": "203.0.113.7"}).status_code == 400
    assert ask(client, "203.0.113.7").status_code == 200


def test_other_addresses_are_not_limited(client):
    """Catches: the limit blocking the website's own calls (drug list, facts, health ping)."""
    from app import store
    import pytest as _p
    mp = _p.MonkeyPatch()
    mp.setattr(store, "get_client", lambda: (_ for _ in ()).throw(RuntimeError("no qdrant in tests")))
    try:
        h = {"X-Forwarded-For": "203.0.113.7"}
        for _ in range(12):                # more than the 10 questions a minute
            assert client.get("/api/health", headers=h).status_code == 200
            assert client.get("/api/drugs?market=pl", headers=h).status_code == 200
            assert client.get("/api/fact?lang=pl", headers=h).status_code == 200
        assert ask(client, "203.0.113.7").status_code == 200
    finally:
        mp.undo()


# ================================================================== 3. settings (.env)

@pytest.mark.parametrize("per_minute, per_day, allowed", [(3, 200, 3), (10, 5, 5), (0, 200, 30), (0, 0, 30)],
                         ids=["3-a-minute", "5-a-day", "minute-off", "all-off"])
def test_the_limits_come_from_the_settings_and_0_switches_them_off(client, monkeypatch, per_minute, per_day, allowed):
    """Catches: numbers hard-coded (the owner can't change them on Render) or 0 not meaning "off"."""
    monkeypatch.setattr(config, "ASK_LIMIT_PER_MINUTE", per_minute)
    monkeypatch.setattr(config, "ASK_LIMIT_PER_DAY", per_day)
    codes = [ask(client, "203.0.113.7").status_code for _ in range(30)]
    assert codes.count(200) == allowed and set(codes[allowed:]) <= {429}


def test_the_settings_are_documented_with_safe_defaults():
    """Catches: a setting the owner can't find in .env.example, or defaults that differ from the agreed "off" (0 / 0) until the IP check on Render."""
    example = (config.ROOT / ".env.example").read_text(encoding="utf-8")
    assert re.search(r"^ASK_LIMIT_PER_MINUTE=0\b", example, re.M)
    assert re.search(r"^ASK_LIMIT_PER_DAY=0\b", example, re.M)
    src = (config.ROOT / "app" / "config.py").read_text(encoding="utf-8")
    assert re.search(r'ASK_LIMIT_PER_MINUTE", "0"', src) and re.search(r'ASK_LIMIT_PER_DAY", "0"', src)


# ================================================================== 4. log, memory, threads

def test_a_block_is_logged_once_a_minute_with_a_hidden_address(client, clock, capsys):
    """Catches: no trace of blocks (the owner can't check the Render setup), a full address in the log, or a
    bot filling the log with one line per refused request."""
    for _ in range(10):
        ask(client, "203.0.113.7")
    capsys.readouterr()
    for _ in range(25):
        ask(client, "203.0.113.7")
    log = capsys.readouterr().out
    lines = [line for line in log.splitlines() if "rate_limited" in line]
    assert len(lines) == 1
    assert "203.0.113.x" in lines[0] and "203.0.113.7" not in log
    clock.wait(61)
    for _ in range(11):
        ask(client, "203.0.113.7")
    assert "rate_limited" in capsys.readouterr().out  # blocked again in a new minute: logged again


def test_memory_stays_bounded_with_many_visitors(clock):
    """Catches: a counter per address kept forever (a bot with many addresses fills the 512 MB server)."""
    lim = limits.RateLimit(max_keys=1000)
    for i in range(5000):
        lim.hit(f"10.{i // 65536}.{i // 256 % 256}.{i % 256}", [(10, 60)])
    assert len(lim) <= 1000


def test_parallel_requests_cannot_slip_past_the_limit(clock):
    """Catches: two requests at the same moment both seeing "9 used" and both getting through (no lock)."""
    import sys
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)            # switch threads as often as possible: makes the race show up
    try:
        for _ in range(20):                # 20 rounds of 50 requests at once
            lim = limits.RateLimit()
            results, start = [], threading.Barrier(50)

            def one():
                start.wait()
                results.append(lim.hit("203.0.113.7", [(10, 60), (200, 86400)]))
            threads = [threading.Thread(target=one) for _ in range(50)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert results.count(0) == 10
    finally:
        sys.setswitchinterval(old)


# ================================================================== 5. the report form uses the real visitor too

def test_report_limit_counts_each_visitor_separately(client):
    """Catches: the report form's 5-per-hour limit using the PROXY's address - on Render all visitors would
    share one counter, and one spammer would block everyone's reports."""
    def report(i):  # 7e: different texts (the same text twice is a duplicate) + the page's timer
        return {"message": f"Wrong answer about Ibuprom {i}", "form_ms": 20_000}
    for i in range(5):
        assert client.post("/api/feedback", json=report(i), headers={"X-Forwarded-For": "203.0.113.7"}).status_code == 200
    report = report(6)
    assert client.post("/api/feedback", json=report, headers={"X-Forwarded-For": "203.0.113.7"}).status_code == 429
    assert client.post("/api/feedback", json=report, headers={"X-Forwarded-For": "198.51.100.20"}).status_code == 200
    assert client.post("/api/feedback", json=report,
                       headers={"X-Forwarded-For": "6.6.6.6, 203.0.113.7"}).status_code == 429  # fake start: no help
