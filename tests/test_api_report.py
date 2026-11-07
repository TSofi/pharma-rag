"""Point 5 of the API hardening (group 7e): spam checks for the "Report a problem" form, no accounts needed.

Rule: clear BOT signs (hidden field filled, sent < 3 s after opening, same visitor + same text again) -> a fake
"sent" ({"ok": true}), no e-mail, the bot learns nothing. Things a REAL person might do (no letters, too many
links, broken screenshot) -> a clear 400. Too many -> 429 (5 an hour per visitor, 90 e-mails a day in total;
Resend's free plan: 100 a day, 3,000 a month). The per-visitor counter forgets old visitors (server memory).

No network: the mail service is a fake; the clock is fake (a "day" costs nothing).
"""
import base64
import re

import pytest
from fastapi.testclient import TestClient

from app import config, feedback, limits, main, spam

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 40
GIF = b"GIF89a" + b"\x00" * 40
WEBP = b"RIFF\x24\x00\x00\x00WEBPVP8 " + b"\x00" * 40


def data_url(kind: str, raw: bytes) -> str:
    return f"data:image/{kind};base64," + base64.b64encode(raw).decode()


class Clock:
    def __init__(self):
        self.t = 5_000_000.0

    def __call__(self):
        return self.t

    def wait(self, seconds):
        self.t += seconds


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(limits, "now", c)
    return c


@pytest.fixture
def mail(monkeypatch, clock):
    """Mail configured, Resend faked. `mail.sent` = payloads; set `mail.fail = True` for a Resend error."""
    class Mail:
        sent, fail = [], False
    m = Mail()
    m.sent = []
    monkeypatch.setattr(config, "RESEND_API_KEY", "key")
    monkeypatch.setattr(config, "FEEDBACK_TO", "owner@example.com")
    monkeypatch.setattr(config, "FEEDBACK_LIMIT_PER_HOUR", 5)
    monkeypatch.setattr(config, "FEEDBACK_LIMIT_PER_DAY", 90)

    def fake_post(url, json, timeout, headers):
        class R:
            status_code = 500 if m.fail else 200
        if not m.fail:
            m.sent.append(json)
        return R()
    monkeypatch.setattr(feedback.httpx, "post", fake_post)
    spam.reset()
    yield m
    spam.reset()


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


def report(client, message="The dose in the answer about Ibuprom looks wrong.", ip="203.0.113.7", **fields):
    body = {"message": message, "form_ms": 20_000, **fields}
    body = {k: v for k, v in body.items() if v is not ...}   # `form_ms=...` = leave the field out
    return client.post("/api/feedback", json=body, headers={"X-Forwarded-For": ip})


# ================================================================== 1. time trap

@pytest.mark.parametrize("form_ms", [3_000, 60_000, 3_600_000])
def test_a_report_written_in_3_s_or_more_is_mailed(client, mail, form_ms):
    """Catches: the time trap refusing real people (writing a report takes more than 3 s)."""
    assert report(client, form_ms=form_ms).json() == {"ok": True}
    assert len(mail.sent) == 1


@pytest.mark.parametrize("form_ms", [0, 500, 2_999, ...], ids=["0", "0.5s", "2.999s", "missing"])
def test_a_report_sent_too_fast_or_without_the_timer_gets_a_fake_ok(client, mail, capsys, form_ms):
    """Catches: a bot (fills and sends in milliseconds, or posts straight to the API without the page's timer)
    getting its report mailed - or learning it was caught (it must see the same "ok" as a real user)."""
    r = report(client, form_ms=form_ms)
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert mail.sent == []
    assert '"reason": "too_fast"' in capsys.readouterr().out


@pytest.mark.parametrize("form_ms", [-1, "soon", 1.5], ids=["negative", "text", "fraction"])
def test_a_broken_timer_value_is_400(client, mail, form_ms):
    """Catches: nonsense in form_ms crashing the check instead of a clear 400."""
    assert report(client, form_ms=form_ms).status_code == 400
    assert mail.sent == []


def test_the_website_sends_how_long_the_form_was_open():
    """Catches: app.js not starting the timer when the form opens (every real report would be dropped)."""
    js = (config.ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    opener = re.search(r'\$\("#reportBtn"\)\.onclick = \(\) => \{(.*?)\};', js, re.S).group(1)
    timer = re.search(r"(\w+) = performance\.now\(\)", opener).group(1)   # started when the form opens
    assert re.search(rf"form_ms: Math\.round\(performance\.now\(\) - {timer}\)", js)


def test_the_hidden_field_still_gives_a_fake_ok_and_is_logged(client, mail, capsys):
    """Catches: the old honeypot lost in the rework."""
    assert report(client, website="http://spam.example").json() == {"ok": True}
    assert mail.sent == []
    assert '"reason": "honeypot"' in capsys.readouterr().out


# ================================================================== 2. the message (a real person can fix it: 400)

@pytest.mark.parametrize("message", ["       ", "\u200b\u200b\u200b\u200b\u200b\u200b", "\n\n\n\t\t\t"],
                         ids=["spaces", "invisible", "newlines"])
def test_an_empty_message_is_400(client, mail, message):
    """Catches: a message of only spaces / invisible characters being e-mailed (it was)."""
    r = report(client, message)
    assert r.status_code == 400 and r.json()["error"] == "EMPTY" and r.json()["field"] == "message"
    assert mail.sent == []


@pytest.mark.parametrize("message", ["?!?!?!?!", "12345 67890", "+++ ---- ****"])
def test_a_message_with_no_letters_is_400(client, mail, message):
    """Catches: "?!?!?!" or "12345" mailed as a problem report."""
    r = report(client, message)
    assert r.status_code == 400 and r.json()["error"] == "NO_LETTERS"
    assert mail.sent == []


def test_a_message_that_is_too_short_after_cleaning_is_400(client, mail):
    """Catches: padding with spaces / invisible characters getting a 2-letter message through."""
    r = report(client, "ok     \u200b\u200b\u200b  ")
    assert r.status_code == 400 and r.json()["error"] == "TOO_SHORT"


@pytest.mark.parametrize("message, status", [
    ("See https://a.example and https://b.example please", 200),
    ("See https://a.example, http://b.example and www.c.example", 400),
    ("BUY https://x.example https://y.example https://z.example https://w.example", 400),
], ids=["2-links-ok", "3-links", "4-links"])
def test_more_than_2_links_is_400_with_a_clear_message(client, mail, message, status):
    """Catches: link spam mailed to the owner - or a real user with 1-2 links (a leaflet URL) refused."""
    r = report(client, message)
    assert r.status_code == status
    if status == 400:
        assert r.json()["error"] == "TOO_MANY_LINKS" and "2" in r.json()["detail"]
        assert mail.sent == []


def test_the_message_is_cleaned_but_keeps_its_line_breaks(client, mail):
    """Catches: invisible characters reaching the e-mail, or a multi-line report glued into one line."""
    assert report(client, "  First line\u200b\r\nSecond\tline\x00\n\n\n\n\nLast line  ").status_code == 200
    assert mail.sent[0]["text"].startswith("First line\nSecond line\n\nLast line\n")


# ================================================================== 3. same visitor + same text = sent once

def test_the_same_visitor_sending_the_same_text_again_gets_a_fake_ok(client, mail, capsys):
    """Catches: a double click on "Send" (or a bot repeating itself) mailing the same report twice."""
    assert report(client).status_code == 200
    assert report(client).json() == {"ok": True}
    assert len(mail.sent) == 1
    assert '"reason": "duplicate"' in capsys.readouterr().out


def test_the_same_text_with_other_spaces_or_capitals_is_still_the_same(client, mail):
    """Catches: a bot dodging the check by changing spaces or letter case."""
    report(client, "The dose in the answer looks wrong.")
    report(client, "  the DOSE in the   answer looks WRONG.  ")
    assert len(mail.sent) == 1


def test_two_people_writing_the_same_words_both_get_through(client, mail):
    """Catches: a real second report lost because someone else described the problem in the same words."""
    assert report(client, "Wrong answer, please check", ip="203.0.113.7").status_code == 200
    assert report(client, "Wrong answer, please check", ip="198.51.100.20").status_code == 200
    assert len(mail.sent) == 2


def test_the_same_text_is_accepted_again_after_24_hours(client, mail, clock):
    """Catches: a visitor never able to send the same words again (the problem may come back next week)."""
    report(client)
    clock.wait(24 * 3600 + 1)
    report(client)
    assert len(mail.sent) == 2


def test_a_report_that_failed_to_send_is_not_remembered(client, mail):
    """Catches: "try again" after a mail-service failure silently dropped as a duplicate."""
    mail.fail = True
    assert report(client).status_code == 502
    mail.fail = False
    assert report(client).status_code == 200
    assert len(mail.sent) == 1


# ================================================================== 4. how many

def test_the_6th_report_in_an_hour_from_one_visitor_is_429(client, mail, clock):
    """Catches: one visitor flooding the owner's inbox; or a limit that never resets."""
    for i in range(5):
        assert report(client, f"Report number {i} about a wrong answer").status_code == 200
    r = report(client, "Report number 6 about a wrong answer")
    assert r.status_code == 429 and r.json()["error"] == "RATE_LIMITED"
    assert "hour" in r.json()["detail"] and 0 < int(r.headers["Retry-After"]) <= 3600
    assert report(client, "Another visitor's report", ip="198.51.100.20").status_code == 200
    clock.wait(3601)
    assert report(client, "Report number 7 about a wrong answer").status_code == 200


def test_the_daily_total_stops_at_the_cap_for_everyone(client, mail, clock, monkeypatch):
    """Catches: many visitors (or a bot with many addresses) using up the free e-mail quota (Resend: 100 a day)."""
    monkeypatch.setattr(config, "FEEDBACK_LIMIT_PER_DAY", 3)
    for i in range(3):
        assert report(client, f"Report {i} from visitor {i}", ip=f"198.51.100.{i}").status_code == 200
    r = report(client, "Report from a new visitor", ip="198.51.100.50")
    assert r.status_code == 429 and "tomorrow" in r.json()["detail"]
    assert int(r.headers["Retry-After"]) > 3600
    assert len(mail.sent) == 3
    clock.wait(24 * 3600 + 1)
    assert report(client, "Report from a new visitor", ip="198.51.100.50").status_code == 200


def test_only_e_mails_really_sent_count_towards_the_daily_cap(client, mail, monkeypatch):
    """Catches: bots (fake "ok") or mail-service failures using up the day's reports of real users."""
    monkeypatch.setattr(config, "FEEDBACK_LIMIT_PER_DAY", 2)
    for i in range(10):
        report(client, f"Spam {i} with a hidden field", ip=f"10.0.0.{i}", website="x")   # bots
        report(client, f"Spam {i} sent too fast", ip=f"10.0.1.{i}", form_ms=10)
    mail.fail = True
    assert report(client, "A real report while the mail service is down", ip="10.0.2.1").status_code == 502
    mail.fail = False
    assert report(client, "A real report", ip="10.0.3.1").status_code == 200
    assert report(client, "Another real report", ip="10.0.3.2").status_code == 200
    assert report(client, "A third real report", ip="10.0.3.3").status_code == 429


@pytest.mark.parametrize("per_hour, per_day, sent", [(0, 90, 8), (5, 0, 5), (0, 0, 8)],
                         ids=["hour-off", "day-off", "all-off"])
def test_the_limits_come_from_the_settings_and_0_switches_them_off(client, mail, monkeypatch, per_hour, per_day, sent):
    """Catches: limits hard-coded (the owner can't change them on Render) or 0 not meaning "off"."""
    monkeypatch.setattr(config, "FEEDBACK_LIMIT_PER_HOUR", per_hour)
    monkeypatch.setattr(config, "FEEDBACK_LIMIT_PER_DAY", per_day)
    for i in range(8):
        report(client, f"Report number {i} about a wrong answer")
    assert len(mail.sent) == sent


def test_the_settings_are_documented_with_the_agreed_defaults():
    """Catches: settings the owner can't find, or defaults other than the agreed 5 an hour / 90 a day."""
    example = (config.ROOT / ".env.example").read_text(encoding="utf-8")
    assert re.search(r"^FEEDBACK_LIMIT_PER_HOUR=5\b", example, re.M)
    assert re.search(r"^FEEDBACK_LIMIT_PER_DAY=90\b", example, re.M)
    src = (config.ROOT / "app" / "config.py").read_text(encoding="utf-8")
    assert 'FEEDBACK_LIMIT_PER_HOUR", "5"' in src and 'FEEDBACK_LIMIT_PER_DAY", "90"' in src


# ================================================================== 5. contact + screenshot

def test_line_breaks_and_hidden_characters_are_removed_from_the_contact(client, mail):
    """Catches: a "contact" with a line break pretending to be extra e-mail lines, or an invisible character
    breaking the reply-to address."""
    report(client, "First report about the dose", contact="me@x.io\nBcc: victim@y.example")
    report(client, "Second report about the dose", contact="  me@x.io\u200b  ")
    first, second = mail.sent
    assert "\nBcc" not in first["text"] and "reply_to" not in first
    assert second["reply_to"] == "me@x.io"


@pytest.mark.parametrize("kind, raw", [("png", PNG), ("jpeg", JPEG), ("jpg", JPEG), ("gif", GIF), ("webp", WEBP)])
def test_a_real_screenshot_is_attached(client, mail, kind, raw):
    """Catches: the new check refusing real PNG / JPEG / GIF / WEBP screenshots."""
    assert report(client, image=data_url(kind, raw)).status_code == 200
    assert mail.sent[0]["attachments"][0]["filename"] == f"screenshot.{kind}"


@pytest.mark.parametrize("kind, raw", [
    ("png", JPEG), ("jpeg", PNG), ("png", b"<html><script>alert(1)</script></html>"), ("gif", b"\x00" * 50),
    ("webp", b"RIFF\x00\x00\x00\x00AVI LIST"), ("png", b""),
], ids=["jpeg-as-png", "png-as-jpeg", "html-as-png", "zeros-as-gif", "avi-as-webp", "empty"])
def test_a_file_that_only_claims_to_be_an_image_is_400(client, mail, kind, raw):
    """Catches: any file attached to the owner's e-mail just because its name says "image/png"."""
    r = report(client, image=data_url(kind, raw))
    assert r.status_code == 400 and r.json()["error"] == "INVALID_IMAGE"
    assert mail.sent == []


# ================================================================== 6. server memory

def test_the_report_counter_forgets_old_visitors(clock):
    """Catches: the counter keeping a line for every address forever (a bot with many addresses could fill
    the 512 MB server) - lines older than an hour are forgotten, and there is a hard maximum."""
    lim = limits.RateLimit(max_keys=1000)
    for i in range(500):
        lim.hit(f"10.0.{i // 256}.{i % 256}", [(5, 3600)])
    assert len(lim) == 500
    clock.wait(3601)
    lim.hit("198.51.100.1", [(5, 3600)])
    assert len(lim) == 1                       # the 500 old visitors are forgotten
    for i in range(5000):
        lim.hit(f"10.1.{i // 256}.{i % 256}", [(5, 3600)])
    assert len(lim) <= 1000                    # never more than the maximum


def test_the_same_text_memory_is_bounded_and_forgets_after_a_day(clock):
    """Catches: the "same text" memory growing forever."""
    for i in range(spam.MAX_REMEMBERED + 500):
        spam.remember_sent(f"10.0.{i // 256}.{i % 256}", f"message {i}")
    assert spam.remembered() <= spam.MAX_REMEMBERED
    clock.wait(24 * 3600 + 1)
    spam.remember_sent("198.51.100.1", "a new message")
    assert spam.remembered() == 1
