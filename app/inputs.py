"""Input checks for the HTTP API (group 7b): language / country codes and the question text.

Used as field types in main.py, e.g. `answer_language: AnswerLanguage = "auto"`. A failed check raises a
pydantic error whose type starts with "app_"; errors.py turns it into a 400 with the code after "app_"
(app_no_letters -> NO_LETTERS) and the message below, prefixed with the field name.
"""
import re
import unicodedata
from typing import Annotated, Callable

from pydantic import AfterValidator, WithJsonSchema
from pydantic_core import PydanticCustomError

from . import config

QUESTION_MIN, QUESTION_MAX = 3, 500

ANSWER_LANGUAGES = ("auto", "en", "uk", "pl", "cs", "es", "fr")   # "auto" = the question's own language
UI_LANGUAGES = ("en", "uk", "pl", "cs", "es", "fr")               # loading-screen facts need a real one
NAMES = {"auto": "same as the question", "en": "English", "uk": "Ukrainian", "pl": "Polish", "cs": "Czech",
         "es": "Spanish", "fr": "French",
         "us": "United States", "ua": "Ukraine", "cz": "Czechia"}  # country codes that differ from language codes
# Easy mix-ups: Ukraine is 'ua' as a country but 'uk' as a language; Czechia 'cz' vs Czech 'cs'.
LANGUAGE_HINTS = {"ua": "uk", "cz": "cs", "gb": "en", "us": "en"}
MARKET_HINTS = {"uk": "ua", "cs": "cz", "en": "us", "usa": "us"}


def _fail(kind: str, message: str):
    return PydanticCustomError(f"app_{kind}", message)


# ---------- codes ----------

def _code_check(allowed: Callable[[], tuple[str, ...]], hints: dict[str, str]):
    """Accepts a known code in any case / with spaces ("EN" -> "en"); otherwise lists the allowed codes and,
    for a known mix-up, says which one was probably meant."""
    def check(value: str) -> str:
        code, ok = value.strip().lower(), allowed()
        if code in ok:
            return code
        message = "must be one of: " + ", ".join(f"'{c}'" for c in ok) + "."
        hint = hints.get(code)
        if hint in ok:
            message += f" Did you mean '{hint}' ({NAMES.get(hint, hint)})?"
        raise _fail("invalid_value", message)
    return check


def _code_type(allowed: Callable[[], tuple[str, ...]], hints: dict[str, str], example: tuple[str, ...]):
    # WithJsonSchema keeps the drop-down list of codes in the Swagger docs (/docs).
    return Annotated[str, AfterValidator(_code_check(allowed, hints)),
                     WithJsonSchema({"type": "string", "enum": list(example)})]


AnswerLanguage = _code_type(lambda: ANSWER_LANGUAGES, LANGUAGE_HINTS, ANSWER_LANGUAGES)
UiLanguage = _code_type(lambda: UI_LANGUAGES, LANGUAGE_HINTS, UI_LANGUAGES)
Market = _code_type(lambda: tuple(config.MARKETS), MARKET_HINTS, tuple(config.MARKETS))


# ---------- the question ----------

def clean_text(text: str) -> str:
    """Text as a person meant it: accents joined (e + ´ -> é), invisible characters (zero-width, BOM, broken
    halves of emoji) removed, control characters (new line, tab, NUL) turned into spaces, spaces collapsed."""
    text = unicodedata.normalize("NFC", text)
    out = []
    for ch in text:
        kind = unicodedata.category(ch)
        if kind == "Cc":                 # control: \n \t \x00 ...
            out.append(" ")
        elif kind not in ("Cf", "Cs"):   # Cf = invisible formatting, Cs = broken character: drop
            out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _question_check(value: str) -> str:
    text = clean_text(value)
    if not text:
        raise _fail("empty", "is empty. Please type a question.")
    if len(text) < QUESTION_MIN:
        raise _fail("too_short", f"is too short (at least {QUESTION_MIN} characters).")
    if len(text) > QUESTION_MAX:
        raise _fail("too_long", f"is too long (at most {QUESTION_MAX} characters).")
    if not any(ch.isalpha() for ch in text):  # a letter in ANY alphabet (Latin, Cyrillic, Chinese, Arabic...)
        raise _fail("no_letters", "must contain words (at least one letter), not only numbers or symbols.")
    return text


Question = Annotated[str, AfterValidator(_question_check),
                    WithJsonSchema({"type": "string", "minLength": QUESTION_MIN, "maxLength": QUESTION_MAX})]


# ---------- the report form (group 7e) ----------

MESSAGE_MIN, MESSAGE_MAX, MAX_LINKS = 5, 3000, 2
_LINK = re.compile(r"(?i)\b(?:https?://|www\.)")


def clean_message(text: str) -> str:
    """Like clean_text, but a report may have several lines: line breaks are kept (at most one empty line in
    a row), everything else is cleaned the same way."""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    out = []
    for ch in text:
        kind = unicodedata.category(ch)
        if ch == "\n":
            out.append(ch)
        elif kind == "Cc":
            out.append(" ")
        elif kind not in ("Cf", "Cs"):
            out.append(ch)
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in "".join(out).split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _message_check(value: str) -> str:
    text = clean_message(value)
    if not text:
        raise _fail("empty", "is empty. Please describe the problem.")
    if len(text) < MESSAGE_MIN:
        raise _fail("too_short", f"is too short (at least {MESSAGE_MIN} characters).")
    if len(text) > MESSAGE_MAX:
        raise _fail("too_long", f"is too long (at most {MESSAGE_MAX} characters).")
    if not any(ch.isalpha() for ch in text):
        raise _fail("no_letters", "must contain words (at least one letter), not only numbers or symbols.")
    if len(_LINK.findall(text)) > MAX_LINKS:
        raise _fail("too_many_links", f"has too many links. Please keep at most {MAX_LINKS}.")
    return text


ReportMessage = Annotated[str, AfterValidator(_message_check),
                          WithJsonSchema({"type": "string", "minLength": MESSAGE_MIN, "maxLength": MESSAGE_MAX})]
# Contact: one line (line breaks / hidden characters removed), so it can't pretend to be extra e-mail lines.
Contact = Annotated[str, AfterValidator(clean_text)]
