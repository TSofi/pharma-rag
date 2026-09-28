"""'Report a problem' form: a visitor's message (+ optional screenshot) is e-mailed to the site owner.
The owner's address and the API key stay on the server; the browser only talks to /api/feedback."""
import base64
import re
import time

import httpx

from . import config

MAX_IMAGE_BYTES = 4 * 1024 * 1024
_recent: dict[str, list[float]] = {}  # ip -> timestamps (tiny in-memory rate limit against spam)


def allowed(ip: str, per_hour: int = 5) -> bool:
    now = time.time()
    hits = [t for t in _recent.get(ip, []) if now - t < 3600]
    _recent[ip] = hits + [now]
    return len(hits) < per_hour


def send(message: str, contact: str, context: str, image: str | None) -> None:
    if not (config.RESEND_API_KEY and config.FEEDBACK_TO):
        raise RuntimeError("feedback e-mail is not configured")
    body = f"{message}\n\n---\nContact: {contact or '(not given)'}\n\n{context}"
    payload = {"from": "PharmaRAG <onboarding@resend.dev>", "to": [config.FEEDBACK_TO],
               "subject": "PharmaRAG: problem report", "text": body}
    if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", contact or ""):
        payload["reply_to"] = contact
    if image:
        m = re.match(r"data:image/(png|jpe?g|webp|gif);base64,(.+)", image, re.S)
        if not m:
            raise ValueError("screenshot must be an image")
        raw = base64.b64decode(m.group(2), validate=True)
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("screenshot is larger than 4 MB")
        payload["attachments"] = [{"filename": f"screenshot.{m.group(1)}", "content": m.group(2)}]
    r = httpx.post("https://api.resend.com/emails", json=payload, timeout=20,
                   headers={"Authorization": f"Bearer {config.RESEND_API_KEY}"})
    if r.status_code >= 400:
        raise RuntimeError(f"mail service error {r.status_code}")
