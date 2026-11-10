"""Point 3 of the API hardening (group 7c): unknown JSON fields -> 400 (strict).

Before, a typo like "answerLanguage" was silently ignored: the answer came in the wrong language and nobody
knew why. Now the JSON body of /api/ask and /api/feedback may contain ONLY the known fields.
Extra parameters in the ADDRESS (?v=2, ?utm_source=...) are still ignored: browsers/tools add them, harmless.

The frontend (app.js) is read by the tests, so a new field added there but not here is caught before deploy.
No network: rag.* and the mail service are fakes.
"""
import re

import pytest
from fastapi.testclient import TestClient

from app import config, feedback, main, rag, spam

OK_Q = "What is metformin?"
ASK_FIELDS = {"question", "top_k", "answer_language", "market"}
FEEDBACK_FIELDS = {"message", "contact", "context", "image", "website",  # website = hidden anti-bot field
                   "form_ms"}  # 7e: how long the form was open (time trap)


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture
def calls(monkeypatch):
    """Everything that got past the door: questions asked and e-mails sent."""
    seen = []
    monkeypatch.setattr(rag, "ask", lambda question, top_k=None, answer_language="auto", market="us":
                        seen.append(("ask", question, answer_language, market)) or {"answer": "ok [1]", "sources": []})
    monkeypatch.setattr(rag, "library", lambda market="us": seen.append(("drugs", market)) or [])
    monkeypatch.setattr(rag, "random_fact", lambda seen_=None, lang="en": seen.append(("fact", lang)) or {})
    monkeypatch.setattr(config, "RESEND_API_KEY", "key")
    monkeypatch.setattr(config, "FEEDBACK_TO", "owner@example.com")

    class R:
        status_code = 200
    monkeypatch.setattr(feedback.httpx, "post", lambda url, json, timeout, headers: seen.append(("mail", json)) or R())
    spam.reset()
    yield seen
    spam.reset()


def frontend_fields(endpoint: str) -> set[str]:
    """Field names app.js puts into JSON.stringify({...}) for this endpoint (shorthand `context` or `key: value`)."""
    js = (config.ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    after = js[js.index(f"/api/{endpoint}`"):]
    start = after.index("JSON.stringify({") + len("JSON.stringify({")
    depth, i = 1, start
    while depth:  # find the matching "}"
        depth += {"{": 1, "}": -1}.get(after[i], 0)
        i += 1
    parts, depth, cur = [], 0, ""
    for ch in after[start:i - 1]:  # split on commas that are not inside (...) / [...] / {...}
        depth += {"(": 1, "[": 1, "{": 1, ")": -1, "]": -1, "}": -1}.get(ch, 0)
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return {re.match(r"\s*(\w+)", p).group(1) for p in parts if p.strip()}


# ---------- unknown fields -> 400 ----------

@pytest.mark.parametrize("extra", [
    {"answerLanguage": "pl"}, {"topK": 5}, {"Market": "pl"}, {"lang": "pl"}, {"admin": True}, {"debug": None},
], ids=["answerLanguage", "topK", "Market", "lang", "admin", "debug-null"])
def test_unknown_field_in_a_question_is_400_naming_it(client, calls, extra):
    """Catches: a typo like "answerLanguage" silently ignored (answer in the wrong language, nobody knows why)."""
    r = client.post("/api/ask", json={"question": OK_Q, **extra})
    assert r.status_code == 400, r.text
    body = r.json()
    name = next(iter(extra))
    assert body["error"] == "UNKNOWN_FIELD" and body["field"] == name
    assert f"'{name}'" in body["detail"] and "not a known field" in body["detail"]
    assert calls == []


@pytest.mark.parametrize("extra", [{"mesage": "typo"}, {"email": "me@x.io"}, {"screenshot": "data:image/png;base64,AAAA"}],
                         ids=["mesage", "email", "screenshot"])
def test_unknown_field_in_a_report_is_400_and_nothing_is_mailed(client, calls, extra):
    """Catches: a report field with the wrong name (e.g. "email" instead of "contact") silently dropped."""
    r = client.post("/api/feedback", json={"message": "Wrong answer about Ibuprom", **extra})
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["error"] == "UNKNOWN_FIELD" and body["field"] == next(iter(extra))
    assert calls == []


def test_misspelled_required_field_is_400_and_never_reaches_the_pipeline(client, calls):
    """Catches: {"Question": ...} (capital Q) reaching the pipeline in any way."""
    r = client.post("/api/ask", json={"Question": OK_Q})
    assert r.status_code == 400
    assert r.json()["error"] in ("MISSING_FIELD", "UNKNOWN_FIELD")
    assert calls == []


# ---------- what must keep working ----------

def test_the_frontend_sends_only_fields_the_server_knows():
    """Catches: a new field added to app.js but not to the server - with strict checking EVERY question or
    report from the website would then fail with 400."""
    assert frontend_fields("ask") <= ASK_FIELDS, frontend_fields("ask") - ASK_FIELDS
    assert frontend_fields("feedback") <= FEEDBACK_FIELDS, frontend_fields("feedback") - FEEDBACK_FIELDS
    assert {"question", "market", "answer_language"} <= frontend_fields("ask")  # the reader itself works


def test_exactly_what_the_frontend_sends_still_works(client, calls):
    """Catches: the strict check refusing the website's own requests."""
    ask = {"question": OK_Q, "market": "pl", "answer_language": "auto"}
    assert set(ask) == frontend_fields("ask")
    assert client.post("/api/ask", json=ask).status_code == 200
    report = {"message": "Wrong answer about Ibuprom", "contact": "me@x.io", "context": "{}", "image": None,
              "website": "", "form_ms": 20_000}
    assert set(report) == frontend_fields("feedback")
    assert client.post("/api/feedback", json=report).status_code == 200
    assert [c[0] for c in calls] == ["ask", "mail"]


def test_the_hidden_anti_bot_field_still_works(client, calls):
    """Catches: the honeypot field "website" treated as unknown (bots would get a 400 = learn it's a trap)."""
    r = client.post("/api/feedback", json={"message": "cheap pills here", "website": "http://spam.example"})
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert calls == []


@pytest.mark.parametrize("url, call", [
    ("/api/fact?lang=pl&v=2", ("fact", "pl")),
    ("/api/drugs?market=pl&utm_source=newsletter", ("drugs", "pl")),
    ("/api/health?ping=1", None),
])
def test_extra_address_parameters_are_still_ignored(client, calls, monkeypatch, url, call):
    """Catches: the strict JSON rule leaking into the address, where browsers and tools add harmless extras."""
    from app import store
    monkeypatch.setattr(store, "get_client", lambda: (_ for _ in ()).throw(RuntimeError("no qdrant in tests")))
    r = client.get(url)
    assert r.status_code != 400 if call is None else r.status_code == 200, r.text  # health: honest 503 (no Qdrant here)
    if call:
        assert calls == [call]


def test_the_accepted_fields_are_exactly_these():
    """Catches: a field added to or removed from the server without anyone noticing (update this list on purpose,
    together with app.js)."""
    assert set(main.AskRequest.model_fields) == ASK_FIELDS
    assert set(main.FeedbackRequest.model_fields) == FEEDBACK_FIELDS
    assert main.AskRequest.model_config.get("extra") == "forbid"
    assert main.FeedbackRequest.model_config.get("extra") == "forbid"
