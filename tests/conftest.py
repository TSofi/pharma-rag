"""Shared test setup: a tiny fake label file so tests never need the real data, the vector DB,
the embedding model or an LLM API key."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config, rag  # noqa: E402

LABELS = [
    {"drug": "metformin", "brands": ["Glucophage"], "url": "https://example.org/metformin", "set_id": "a",
     "effective_time": "20251101", "sections": [
         {"section": "Indications and Usage",
          "text": "1 INDICATIONS AND USAGE Metformin is indicated as an adjunct to diet and exercise to improve "
                  "glycemic control in adults with type 2 diabetes mellitus ( 1 )."}]},
    {"drug": "ibuprofen", "brands": ["Advil", "Nurofen"], "url": "https://example.org/ibuprofen", "set_id": "b",
     "effective_time": "20250101", "sections": [
         {"section": "Boxed Warning", "text": "WARNING: RISK OF SERIOUS CARDIOVASCULAR EVENTS"},
         {"section": "Mechanism of Action",
          "text": "12.1 Mechanism of Action Ibuprofen has analgesic, anti-inflammatory and antipyretic properties "
                  "[see Clinical Pharmacology (12.2)]."}]},
    {"drug": "naproxen", "brands": ["Aleve"], "url": "https://example.org/naproxen", "set_id": "c",
     "effective_time": "20250101", "sections": []},
]


@pytest.fixture(autouse=True)
def fake_labels(tmp_path, monkeypatch):
    path = tmp_path / "labels.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in LABELS), encoding="utf-8")
    monkeypatch.setattr(config, "DATA_FILE", path)
    for fn in (rag.drug_aliases, rag.drug_brands):
        fn.cache_clear()
    yield
    for fn in (rag.drug_aliases, rag.drug_brands):
        fn.cache_clear()
