"""Answer cache: the same question again within ANSWER_CACHE_TTL_S (15 min) gets the saved answer, without
calling Qdrant or the LLM (faster for the user, saves the free LLM quota).

Key = normalized question (case and extra spaces don't matter) + market + requested answer language + top_k.
Only CLEAN answers are saved (`skip_reason`). An allowlist: only the known complete answer modes
(CACHEABLE_MODES) -- anything new or unexpected is not saved by accident. And never:
  - emergency answers (red card, and the soft red card shown while the LLM is down),
  - errors (they never reach the cache: the API raises before storing),
  - answers whose meaning check could not run (judge LLM down),
  - answers with an "issue" (a step failed or a safety check rejected the answer, observability.ISSUE_EVENTS):
    a second try a minute later may well give the full answer.
Lives in RAM only (gone after a restart / deploy), at most ANSWER_CACHE_MAX answers, least recently used first out.
"""
import copy
import threading
import time
from collections import OrderedDict

from . import config
from .observability import normalize_question

now = time.time  # replaced by a fake clock in tests


# violet (from the leaflet), orange (general answer), joke, plain "not found". NOT "emergency".
CACHEABLE_MODES = frozenset({"sourced", "general", "playful", "none"})


def skip_reason(result: dict, issues: list[str]) -> str | None:
    """Why this answer must NOT be cached (None = it's clean, cache it)."""
    if result.get("mode") == "emergency":
        return "emergency"
    if result.get("mode") not in CACHEABLE_MODES:
        return "unknown_mode"
    if result.get("llm_unavailable"):
        return "llm_unavailable"
    if result.get("meaning_check") == "unavailable":
        return "meaning_check_unavailable"
    if issues:
        return "issues"
    return None


class AnswerCache:
    def __init__(self):
        self._items: OrderedDict[tuple, tuple[float, dict]] = OrderedDict()  # key -> (saved at, answer)
        self._lock = threading.Lock()  # questions are answered in parallel threads

    @staticmethod
    def enabled() -> bool:
        return config.ANSWER_CACHE_TTL_S > 0 and config.ANSWER_CACHE_MAX > 0

    @staticmethod
    def key(question: str, market: str, answer_language: str, top_k: int | None) -> tuple:
        return normalize_question(question), market, answer_language, top_k or config.TOP_K

    def get(self, key: tuple) -> dict | None:
        """A copy of the saved answer, or None (not saved / older than the TTL / cache off)."""
        if not self.enabled():
            return None
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            saved_at, answer = item
            if now() - saved_at >= config.ANSWER_CACHE_TTL_S:
                del self._items[key]
                return None
            self._items.move_to_end(key)  # recently used = kept longest
            return copy.deepcopy(answer)  # the caller may change its copy; the saved one stays as it was

    def put(self, key: tuple, answer: dict) -> None:
        if not self.enabled():
            return
        saved = copy.deepcopy({k: v for k, v in answer.items() if k not in ("request_id", "timings", "cached")})
        with self._lock:
            self._items[key] = (now(), saved)
            self._items.move_to_end(key)
            while len(self._items) > config.ANSWER_CACHE_MAX:
                self._items.popitem(last=False)  # the least recently used

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


answers = AnswerCache()
