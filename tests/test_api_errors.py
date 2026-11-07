"""Point 1 of the API hardening: every error has an exact HTTP number and the same JSON shape.

    {"detail": "<short readable text>", "error": "<CODE>", ...}

400 = the request is broken/wrong, 404/405 = wrong address/method, 413 = request too big,
429 = too many requests, 500 = bug on our side, 502 = a service we depend on failed,
503 = we are busy / not set up. The input is never sent back, internal details never leave the server.

No network: rag.ask / rag.library and the mail service (Resend) are fakes.
"""
import httpx
import pytest
from fastapi.testclient import TestClient

from app import config, feedback, main, rag, spam

SECRET = "https://secret-cluster.qdrant.io key=abc123"  # an "internal detail" that must never reach the user


@pytest.fixture
def client():
    # raise_server_exceptions=False: we want to see what the USER gets, not the Python exception.
    return TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture
def asked(monkeypatch):
    """Fake rag.ask that records every call (so a test can prove the pipeline was NOT reached)."""
    calls = []

    def fake_ask(question, top_k=None, answer_language="auto", market="us"):
        calls.append({"question": question, "top_k": top_k, "answer_language": answer_language, "market": market})
        return {"answer": "ok [1]", "found": True, "sources": []}
    monkeypatch.setattr(rag, "ask", fake_ask)
    return calls


@pytest.fixture
def mail(monkeypatch):
    """Feedback e-mail configured, Resend faked. `mail.sent` = payloads; set `mail.fail` to an exception or code."""
    class Mail:
        sent, fail = [], None
    m = Mail()
    m.sent = []
    monkeypatch.setattr(config, "RESEND_API_KEY", "re_secret_key")
    monkeypatch.setattr(config, "FEEDBACK_TO", "owner@example.com")

    def fake_post(url, json, timeout, headers):
        if isinstance(m.fail, Exception):
            raise m.fail

        class R:
            status_code = m.fail or 200
        m.sent.append(json)
        return R()
    monkeypatch.setattr(feedback.httpx, "post", fake_post)
    spam.reset()
    yield m
    spam.reset()


def shape_ok(r):
    """The one error shape the frontend relies on: JSON, `detail` is a STRING, `error` is a code."""
    assert r.headers["content-type"].startswith("application/json"), r.text
    body = r.json()
    assert isinstance(body["detail"], str) and body["detail"], body
    assert isinstance(body["error"], str) and body["error"].isupper(), body
    return body


# ---------- 400: broken or wrong request (was 422 with a list the UI showed as "[object Object]") ----------

@pytest.mark.parametrize("path", ["/api/ask", "/api/feedback"])
def test_broken_json_is_400_with_a_clear_message(client, asked, path):
    """Catches: invalid JSON answered with 422 and a list instead of a readable 400."""
    r = client.post(path, content=b'{"question": ', headers={"content-type": "application/json"})
    assert r.status_code == 400
    body = shape_ok(r)
    assert body["error"] == "INVALID_JSON" and "JSON" in body["detail"]
    assert asked == []


@pytest.mark.parametrize("kwargs", [
    {},                                                                            # no body at all
    {"json": ["What is metformin?"]},                                              # array
    {"json": "What is metformin?"},                                                # plain string
    {"json": 42},                                                                  # number
    {"content": b"null", "headers": {"content-type": "application/json"}},         # null
    {"content": b"What is metformin?", "headers": {"content-type": "text/plain"}},  # not JSON at all
], ids=["no-body", "array", "string", "number", "null", "text-plain"])
def test_body_that_is_not_a_json_object_is_400(client, asked, kwargs):
    """Catches: a missing / non-object body answered with 422 or reaching the pipeline."""
    r = client.post("/api/ask", **kwargs)
    assert r.status_code == 400
    assert shape_ok(r)["error"] == "INVALID_BODY"
    assert asked == []


@pytest.mark.parametrize("payload, field, code, words", [
    ({}, "question", "MISSING_FIELD", ["question", "required"]),
    ({"question": ""}, "question", "EMPTY", ["question", "empty"]),   # 7b: was TOO_SHORT
    ({"question": "ab"}, "question", "TOO_SHORT", ["question", "3"]),
    ({"question": "a" * 501}, "question", "TOO_LONG", ["question", "500"]),
    ({"question": 12345}, "question", "WRONG_TYPE", ["question", "text"]),
    ({"question": "What is metformin?", "top_k": 0}, "top_k", "OUT_OF_RANGE", ["top_k", "1"]),
    ({"question": "What is metformin?", "top_k": 16}, "top_k", "OUT_OF_RANGE", ["top_k", "15"]),
    ({"question": "What is metformin?", "top_k": "many"}, "top_k", "WRONG_TYPE", ["top_k", "number"]),
    ({"question": "What is metformin?", "answer_language": "de"}, "answer_language", "INVALID_VALUE",
     ["answer_language", "auto", "en", "uk", "pl", "cs", "es", "fr"]),
    ({"question": "What is metformin?", "market": "xx"}, "market", "INVALID_VALUE",
     ["market", "us", "pl", "ua", "cz", "es", "fr"]),
], ids=["missing", "empty", "too-short", "too-long", "not-text", "top_k-0", "top_k-16", "top_k-word",
        "bad-language", "bad-market"])
def test_wrong_field_is_400_naming_the_field(client, asked, payload, field, code, words):
    """Catches: a wrong field answered with 422, or a message that doesn't say WHICH field and WHY."""
    r = client.post("/api/ask", json=payload)
    assert r.status_code == 400
    body = shape_ok(r)
    assert body["error"] == code and body["field"] == field
    for w in words:
        assert w in body["detail"], (w, body["detail"])
    assert asked == []


@pytest.mark.parametrize("url, field", [("/api/fact?lang=de", "lang"), ("/api/drugs?market=xx", "market")])
def test_bad_query_parameter_is_400(client, url, field):
    """Catches: a bad ?lang= / ?market= in the address answered with 422."""
    r = client.get(url)
    assert r.status_code == 400
    body = shape_ok(r)
    assert body["error"] == "INVALID_VALUE" and body["field"] == field


def test_error_never_sends_the_input_back(client, asked):
    """Catches: the 422 habit of echoing the whole input (a 6 MB screenshot came back as 6 MB)."""
    r = client.post("/api/ask", json={"question": "SECRET-INPUT-" * 60})
    assert r.status_code == 400
    assert "SECRET-INPUT" not in r.text and len(r.content) < 500


def test_a_valid_question_still_reaches_the_pipeline_unchanged(client, asked):
    """Catches: the new error handling getting in the way of a normal question."""
    r = client.post("/api/ask", json={"question": "What is metformin?", "top_k": 5,
                                      "answer_language": "pl", "market": "pl"})
    assert r.status_code == 200 and r.json()["answer"] == "ok [1]"
    assert asked == [{"question": "What is metformin?", "top_k": 5, "answer_language": "pl", "market": "pl"}]


# ---------- 413: request too big ----------

def test_too_big_question_request_is_413_and_never_reaches_the_pipeline(client, asked):
    """Catches: a huge body being read and parsed in full on a 512 MB server."""
    r = client.post("/api/ask", json={"question": "What is metformin?", "padding": "x" * 50_000})
    assert r.status_code == 413
    assert shape_ok(r)["error"] == "TOO_LARGE"
    assert asked == []


def test_too_big_report_is_413_with_a_short_answer(client, mail):
    """Catches: an oversized screenshot being echoed back (6 MB in, 6 MB out) or e-mailed."""
    r = client.post("/api/feedback", json={"message": "hello there", "image": "x" * 7_000_000})
    assert r.status_code == 413
    assert shape_ok(r)["error"] == "TOO_LARGE" and len(r.content) < 500
    assert mail.sent == []


def test_size_limit_also_works_without_a_content_length_header(client, asked):
    """Catches: a client that streams the body (no Content-Length) getting around the limit."""
    def chunks():
        yield b'{"question": "What is metformin?", "padding": "'
        for _ in range(10):
            yield b"x" * 10_000
        yield b'"}'
    r = client.post("/api/ask", content=chunks(), headers={"content-type": "application/json"})
    assert r.status_code == 413
    assert asked == []


def test_a_normal_4_mb_screenshot_is_not_blocked_by_the_size_limit(client, mail):
    """Catches: a size limit so low that a real screenshot (max 4 MB -> ~5.6 MB as text) can't be sent."""
    import base64
    raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * (4 * 1024 * 1024 - 8)   # a real PNG start, exactly 4 MB (the app's limit)
    png = "data:image/png;base64," + base64.b64encode(raw).decode()   # ~5.6 MB as text
    r = client.post("/api/feedback", json={"message": "Wrong answer about Ibuprom", "image": png, "form_ms": 20_000})
    assert r.status_code == 200, r.text[:200]
    assert len(mail.sent) == 1


# ---------- 429: too many ----------

def test_too_many_reports_is_429_with_a_wait_time(client, mail):
    """Catches: the rate-limit answer without the shared shape or without telling the user how long to wait."""
    for i in range(5):  # 7e: different texts (the same text twice is a duplicate) + form_ms
        assert client.post("/api/feedback", json={"message": f"hello there {i}", "form_ms": 20_000}).status_code == 200
    r = client.post("/api/feedback", json={"message": "hello there 6", "form_ms": 20_000})
    assert r.status_code == 429
    body = shape_ok(r)
    assert body["error"] == "RATE_LIMITED" and "hour" in body["detail"]
    assert int(r.headers["Retry-After"]) > 0
    assert len(mail.sent) == 5


# ---------- 404 / 405: wrong address / wrong method ----------

def test_unknown_api_address_is_a_json_404(client):
    """Catches: an unknown /api/... address answered in a different format than every other error."""
    r = client.get("/api/does-not-exist")
    assert r.status_code == 404
    assert shape_ok(r)["error"] == "NOT_FOUND"


@pytest.mark.parametrize("method, path, allowed", [
    ("GET", "/api/ask", "POST"), ("DELETE", "/api/ask", "POST"), ("GET", "/api/feedback", "POST"),
    ("POST", "/api/health", "GET"), ("PUT", "/api/drugs", "GET"),
])
def test_wrong_method_is_405_and_says_which_one_is_right(client, method, path, allowed):
    """Catches: GET /api/ask answered "404 not found" although the address exists (it needs POST), or a
    405 naming the wrong method (the website part answering "use GET" for /api/ask)."""
    r = client.request(method, path)
    assert r.status_code == 405
    assert shape_ok(r)["error"] == "METHOD_NOT_ALLOWED"
    assert r.headers["Allow"] == allowed


@pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
def test_unknown_api_address_is_404_whatever_the_method(client, method):
    """Catches: POST /api/nothing-here answered "405 wrong method" although the address doesn't exist."""
    r = client.request(method, "/api/nothing-here")
    assert r.status_code == 404
    assert shape_ok(r)["error"] == "NOT_FOUND"


def test_the_website_itself_is_still_served(client):
    """Catches: the new error handling swallowing the frontend pages."""
    r = client.get("/")
    assert r.status_code == 200 and "<html" in r.text.lower()


# ---------- 500: a bug in our code ----------

def test_a_crash_is_a_json_500_without_details_and_the_details_go_to_the_log(client, monkeypatch, capsys):
    """Catches: a crash answered as plain text (the UI can't read it) or showing internal details."""
    def crash(*_a, **_k):
        raise RuntimeError(SECRET)
    monkeypatch.setattr(rag, "library", crash)
    r = client.get("/api/drugs?market=pl")
    assert r.status_code == 500
    body = shape_ok(r)
    assert body["error"] == "INTERNAL_ERROR" and body["request_id"]
    assert "secret" not in r.text.lower() and "RuntimeError" not in r.text
    log = capsys.readouterr().out
    assert SECRET in log and body["request_id"] in log  # the owner can find the full error by its id


def test_error_answers_keep_the_cors_header(client, monkeypatch):
    """Catches: errors without the CORS header - the browser then hides them and the UI shows "API unreachable"."""
    monkeypatch.setattr(rag, "library", lambda *_a, **_k: 1 / 0)
    origin = {"Origin": "https://pharma-rag-nine.vercel.app"}
    responses = [
        client.get("/api/drugs", headers=origin),                                         # 500
        client.post("/api/ask", json={"question": "x" * 20_000}, headers=origin),          # 413
        client.post("/api/ask", json={"question": ""}, headers=origin),                    # 400
    ]
    assert [r.status_code for r in responses] == [500, 413, 400]
    for r in responses:
        assert "access-control-allow-origin" in r.headers, r.status_code


# ---------- 502 / 503: a service we depend on ----------

def test_search_failure_is_502_without_internal_details(client, monkeypatch, capsys):
    """Catches: "RuntimeError: <Qdrant URL / key>" being shown to every visitor."""
    def down(*_a, **_k):
        raise RuntimeError(SECRET)
    monkeypatch.setattr(rag, "ask", down)
    r = client.post("/api/ask", json={"question": "What is metformin?"})
    assert r.status_code == 502
    body = shape_ok(r)
    assert body["error"] == "UPSTREAM_ERROR"
    assert "secret" not in r.text.lower() and "RuntimeError" not in r.text
    assert SECRET in capsys.readouterr().out


@pytest.mark.parametrize("raw", ["503 UNAVAILABLE. The model is overloaded.", "429 RESOURCE_EXHAUSTED quota",
                                 "Qdrant: service overloaded, try later"])
def test_a_busy_service_is_a_clean_503_the_ui_offers_to_retry(client, monkeypatch, raw):
    """Catches: "busy" errors turned into a 502 (the UI then says "something went wrong", no retry hint),
    or the raw provider text reaching the user."""
    def busy(*_a, **_k):
        raise RuntimeError(f"{raw} ({SECRET})")
    monkeypatch.setattr(rag, "ask", busy)
    r = client.post("/api/ask", json={"question": "What is metformin?"})
    assert r.status_code == 503 and int(r.headers["Retry-After"]) > 0
    body = shape_ok(r)
    assert body["error"] == "SERVICE_BUSY" and "503" in body["detail"]  # "503" = the UI's "busy" signal
    assert "secret" not in r.text.lower() and raw not in r.text


def test_a_number_inside_another_number_is_not_a_busy_signal():
    """Catches: "port 15030" or "id 4290" being read as 503 / 429."""
    from app import errors
    assert not errors.looks_busy(RuntimeError("connection refused on port 15030, id 4290"))
    assert errors.looks_busy(RuntimeError("HTTP 503 from upstream"))


def test_llm_busy_stays_a_503_the_ui_recognises(client, monkeypatch):
    """Catches: the new handling changing the "busy, try again" answer the UI and frozen tests rely on."""
    from app import llm

    def busy(*_a, **_k):
        raise llm.LLMUnavailable("all down", retry_after_s=20)
    monkeypatch.setattr(rag, "ask", busy)
    r = client.post("/api/ask", json={"question": "What is metformin?"})
    assert r.status_code == 503 and r.headers["Retry-After"] == "20"
    body = shape_ok(r)
    assert body["error"] == "LLM_UNAVAILABLE" and body["detail"].startswith("LLM_UNAVAILABLE")


def test_report_form_not_set_up_is_503(client, monkeypatch):
    """Catches: a missing e-mail setup on OUR side reported as 502 "the other service failed"."""
    monkeypatch.setattr(config, "RESEND_API_KEY", "")
    spam.reset()
    r = client.post("/api/feedback", json={"message": "hello there"})
    assert r.status_code == 503
    assert shape_ok(r)["error"] == "FEEDBACK_UNAVAILABLE"


@pytest.mark.parametrize("failure", [500, httpx.ConnectError(f"cannot reach api.resend.com ({SECRET})")],
                         ids=["resend-500", "no-connection"])
def test_mail_service_failure_is_502_without_details(client, mail, failure):
    """Catches: the mail service's error text (or our key) being shown to the visitor."""
    mail.fail = failure
    r = client.post("/api/feedback", json={"message": "hello there", "form_ms": 20_000})
    assert r.status_code == 502
    body = shape_ok(r)
    assert body["error"] == "UPSTREAM_ERROR"
    assert "secret" not in r.text.lower() and "re_secret_key" not in r.text


def test_a_screenshot_that_is_not_an_image_is_400(client, mail):
    """Catches: a bad screenshot reported as a server problem instead of the user's input."""
    r = client.post("/api/feedback", json={"message": "hello there", "image": "data:text/html;base64,AAAA",
                                           "form_ms": 20_000})
    assert r.status_code == 400
    assert shape_ok(r)["error"] == "INVALID_IMAGE"
    assert mail.sent == []
