"""Group 1: leaflet parsing (scripts/doc_text.py).

These functions turn a downloaded leaflet into (section label, text) pairs. Every chunk in Qdrant comes
from here, so a broken regex silently degrades the whole index. Each test below names the mistake it
catches. Sample leaflets live in tests/fixtures/leaflets/ (short, realistic excerpts, one per country).
"""
import re
import unicodedata
from email.message import EmailMessage
from pathlib import Path
from unittest import mock

import pytest

from scripts import doc_text
from scripts.doc_text import (PL_SECTIONS, _tidy, any_text, fr_notice, html_text, leaflet_part, mht_text,
                              sections_pl, sections_ua)
from scripts.ingest import chunk_text

FIXTURES = Path(__file__).parent / "fixtures" / "leaflets"
QRD_LABELS = [label for _, _, label in PL_SECTIONS]  # the 6 canonical section names, same in every language


def leaflet(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# ---------- EU leaflets (QRD template): PL / CZ / ES / FR ----------
@pytest.mark.parametrize("lang, file, alcohol_word", [
    ("pl", "pl_ibuprofen.txt", "alkoholu"),
    ("cz", "cz_ibuprofen.txt", "alkohol"),
    ("es", "es_ibuprofeno.txt", "alcohol"),
    ("fr", "fr_ibuprofene.txt", "alcool"),
])
def test_qrd_leaflet_splits_into_the_six_canonical_sections(lang, file, alcohol_word):
    """Catches: a heading regex in QRD[lang] stops matching -> sections merge or vanish.
    Also guarantees the labels are identical across languages (the UI and prompts rely on that)."""
    secs = sections_pl(leaflet(file), lang)
    labels = [label for label, _ in secs]
    assert labels == ["Introduction"] + QRD_LABELS, labels   # title + TOC become the "Introduction" block
    by_label = dict(secs)
    assert alcohol_word in by_label[QRD_LABELS[1]].lower()          # interactions live in section 2
    assert "12" in by_label[QRD_LABELS[2]]                           # dosage (age 12+) lives in section 3
    assert "25" in by_label[QRD_LABELS[4]]                           # storage temperature in section 5


@pytest.mark.parametrize("lang, file, toc_marker", [
    ("pl", "pl_ibuprofen.txt", "Spis treści"),
    ("cz", "cz_ibuprofen.txt", "Co naleznete"),
    ("es", "es_ibuprofeno.txt", "Contenido del prospecto"),
    ("fr", "fr_ibuprofene.txt", "Que contient cette notice"),
])
def test_table_of_contents_is_not_mistaken_for_the_headings(lang, file, toc_marker):
    """Catches: the splitter takes the FIRST "1. Co to jest" (inside the table of contents) instead of
    the real heading, so section 1 would be the whole TOC and section 6 the whole leaflet."""
    secs = dict(sections_pl(leaflet(file), lang))
    assert toc_marker not in secs[QRD_LABELS[0]]
    assert toc_marker in secs.get("Introduction", "")            # the TOC ends up in the intro block
    assert len(secs[QRD_LABELS[5]]) < 400                            # last section is really just section 6


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _headings_in_caps(text: str, keep_accents: bool) -> str:
    """Rewrite the six numbered heading lines the way many French/Spanish leaflets print them:
    CAPITALS, usually without accents."""
    def caps(m):
        line = m.group(0).upper()
        return line if keep_accents else _strip_accents(line)
    return re.sub(r"(?m)^[1-6]\. .*$", caps, text)


@pytest.mark.parametrize("lang, file", [("fr", "fr_ibuprofene.txt"), ("es", "es_ibuprofeno.txt"),
                                        ("cz", "cz_ibuprofen.txt"), ("pl", "pl_ibuprofen.txt")])
@pytest.mark.parametrize("keep_accents", [True, False])
def test_headings_in_capitals_with_or_without_accents_are_found(lang, file, keep_accents):
    """Catches: "4. QUELS SONT LES EFFETS INDESIRABLES" (capitals, accents dropped -- the usual French
    typing habit, same in Spanish "COMO TOMAR") not matching "indésirables", so the side-effects text
    is glued to the dosage section under the wrong label."""
    text = _headings_in_caps(leaflet(file), keep_accents)
    secs = sections_pl(text, lang)
    assert [label for label, _ in secs] == ["Introduction"] + QRD_LABELS, [l for l, _ in secs]
    assert "25" in dict(secs)[QRD_LABELS[4]]                        # storage section really split off


@pytest.mark.parametrize("sep", [". ", " - ", " – ", ") ", " "])
def test_heading_number_separator_variants_are_accepted(sep):
    """Catches: a leaflet numbering its headings "4 - Quels sont..." or "4) Quels sont..." instead of
    "4. Quels sont..." losing every section."""
    text = re.sub(r"(?m)^([1-6])\. ", lambda m: m.group(1) + sep, leaflet("fr_ibuprofene.txt"))
    assert [label for label, _ in sections_pl(text, "fr")] == ["Introduction"] + QRD_LABELS


def test_accent_folding_keeps_section_text_intact():
    """Catches: the accent-blind matching being done on a modified copy of the leaflet, so the saved
    section text loses its accents (the LLM would see "indesirables", the UI would show mangled French)."""
    secs = dict(sections_pl(leaflet("fr_ibuprofene.txt"), "fr"))
    assert "indésirables" in secs[QRD_LABELS[3]] and "température" in secs[QRD_LABELS[4]]


def test_leaflet_with_fewer_than_three_headings_is_kept_as_one_block():
    """Catches: an unusual leaflet (no numbered headings) being thrown away instead of indexed whole."""
    text = "Ulotka informacyjna\n" + "Lek stosuje się w bólu. " * 20
    assert [label for label, _ in sections_pl(text, "pl")] == ["Leaflet"]


def test_empty_or_whitespace_text_gives_no_sections():
    assert sections_pl("", "pl") == []
    assert sections_pl("   \n\n  ", "pl") == []
    assert sections_ua("") == []


def test_tiny_sections_are_dropped():
    """Catches: 1-line sections (e.g. "5. Jak przechowywać\nPatrz opakowanie.") becoming useless chunks."""
    text = leaflet("pl_ibuprofen.txt")
    text = text.replace("5. Jak przechowywać lek Nurofen Forte\nPrzechowywać w temperaturze poniżej 25°C, "
                        "w miejscu niewidocznym i niedostępnym dla dzieci.", "5. Jak przechowywać\nPatrz pudełko.")
    labels = [label for label, _ in sections_pl(text, "pl")]
    assert QRD_LABELS[4] not in labels and len(labels) == 6   # Introduction + 5 real sections


# ---------- Ukrainian instructions (MOZ register) ----------
def test_ua_instruction_splits_into_labelled_sections_in_order():
    """Catches: a Ukrainian heading regex drifting, so e.g. Interactions merges into Contraindications."""
    secs = sections_ua(leaflet("ua_ibuprofen.txt"))
    labels = [label for label, _ in secs]
    expected = ["Composition", "Dosage form", "Pharmacotherapeutic group", "Pharmacological properties",
                "Indications", "Contraindications", "Interactions", "Warnings and precautions",
                "Pregnancy and breastfeeding", "Driving and using machines", "Dosage and administration",
                "Children", "Overdose", "Side effects", "Storage", "Package", "Prescription status"]
    assert labels == expected, labels
    by_label = dict(secs)
    assert "Алкоголь" in by_label["Interactions"]
    assert "1200 мг" in by_label["Dosage and administration"]
    assert "Клінічні характеристики" not in by_label["Indications"]   # the unlabelled umbrella heading is skipped


def test_ua_manufacturer_and_unlabelled_headings_are_not_indexed():
    """Catches: company addresses (Manufacturer) or the "Клінічні характеристики" umbrella heading
    getting their own chunks -- they never answer a question."""
    labels = [label for label, _ in sections_ua(leaflet("ua_ibuprofen.txt"))]
    assert "Manufacturer" not in labels and None not in labels
    assert "Реккітт" not in " ".join(t for _, t in sections_ua(leaflet("ua_ibuprofen.txt")))


def test_ua_text_with_few_headings_is_kept_as_one_block():
    text = "Показання. Біль.\nПротипоказання. Алергія.\n" + "Текст інструкції. " * 30
    assert [label for label, _ in sections_ua(text)] == ["Instruction"]


# ---------- EMA product information: SmPC + labelling + leaflet(s) ----------
EMA_HEAD = {"pl": ("B. ULOTKA DLA PACJENTA", "Ulotka dołączona do opakowania: informacja dla pacjenta"),
            "cz": ("B. PŘÍBALOVÁ INFORMACE", "Příbalová informace: informace pro pacienta"),
            "es": ("B. PROSPECTO", "Prospecto: información para el usuario"),
            "fr": ("B. NOTICE", "Notice : information de l'utilisateur")}


@pytest.mark.parametrize("lang", ["pl", "cz", "es", "fr"])
def test_ema_pdf_keeps_only_the_first_patient_leaflet(lang):
    """Catches: indexing the 60-page SmPC (section "4.5 Interakcje" etc.) instead of the patient leaflet,
    or indexing the same leaflet twice when the PDF has one leaflet per strength."""
    head, title = EMA_HEAD[lang]
    smpc = "ANEKS I\nCHARAKTERYSTYKA PRODUKTU LECZNICZEGO\n4.5 Interakcje\n" + "SmPC text. " * 50
    first = f"{title}\nEliquis 2,5 mg tabletki\n" + "FIRST leaflet text. " * 30
    second = f"{title}\nEliquis 5 mg tabletki\n" + "SECOND leaflet text. " * 30
    text = f"{smpc}\n{head}\n{first}\n{second}"
    out = leaflet_part(text, lang).strip()
    assert out.startswith(title) and "2,5 mg" in out
    assert "SmPC text" not in out and "SECOND" not in out


def test_text_without_ema_heading_is_returned_unchanged():
    """Catches: a national (non-EMA) leaflet being truncated because leaflet_part is applied to it."""
    text = leaflet("pl_ibuprofen.txt")
    assert leaflet_part(text, "pl") == text


# ---------- French BDPM web page: navigation + SmPC + notice ----------
def test_fr_bdpm_page_keeps_only_the_notice():
    """Catches: the French index containing the site menu, the SmPC or the page footer."""
    page = ("Base de Données Publique des Médicaments\nAccueil Lien\nRésumé des caractéristiques du produit\n"
            "ANSM - Mis à jour le 10/01/2020\n4.5. Interactions\n" + "RCP text. " * 20 + "\n"
            "Notice patient\nANSM - Mis à jour le 12/03/2021\n" + leaflet("fr_ibuprofene.txt") + "\n"
            "Redirection vers le haut de page\nPlan du site Lien\nMentions légales Lien\nfooter text")
    notice = fr_notice(page)
    assert notice.startswith("ANSM - Mis à jour le 12/03/2021")
    assert "RCP text" not in notice and "Plan du site" not in notice and "haut de page" not in notice
    # and the whole pipeline for FR uses it:
    labels = [label for label, _ in sections_pl(page, "fr")]
    assert labels == ["Introduction"] + QRD_LABELS


def test_fr_notice_without_marker_is_empty():
    """Catches: a BDPM page with no notice (some products have only an SmPC) being indexed as a blob."""
    assert fr_notice("Base de Données Publique des Médicaments\nno notice here") == ""
    assert sections_pl("Base de Données Publique des Médicaments\nno notice here", "fr") == []


# ---------- raw formats: HTML, MHT, dispatch, tidy ----------
def test_html_text_drops_style_script_and_keeps_table_cells():
    """Catches: CSS/JS ending up in chunks, or table cells glued together ("200 mg400 mg")."""
    markup = ("<html><head><style>p{color:red}</style><script>var x=1;</script></head><body>"
              "<h1>Skład</h1><p>Ibuprofen&nbsp;200&nbsp;mg</p>"
              "<table><tr><td>Dawka</td><td>400 mg</td></tr></table>"
              "<!--[if gte mso 9]><xml>office junk</xml><![endif]--></body></html>")
    out = html_text(markup)
    assert "color" not in out and "var x" not in out and "office junk" not in out
    assert "Skład" in out and "Ibuprofen 200 mg" in out and "Dawka | 400 mg" in out


def _mht(parts: list[tuple[str, bytes, str | None]]) -> bytes:
    """Build a Word-style .mht (multipart/related) file. parts = [(filename, html bytes, charset or None)]."""
    msg = EmailMessage()
    msg["MIME-Version"] = "1.0"
    msg.make_related()
    for name, body, charset in parts:
        part = EmailMessage()
        part.set_content(body, maintype="text", subtype="html", cte="base64")
        if charset:
            part.set_param("charset", charset, header="Content-Type")
        part.add_header("Content-Location", name)
        msg.attach(part)
    return msg.as_bytes()


def test_mht_text_decodes_cp1251_and_picks_the_main_document():
    """Catches: Ukrainian instructions decoded as UTF-8 (mojibake) or the tiny header.htm chosen
    instead of the real document."""
    main_html = ("<html><head><meta charset=\"windows-1251\"></head><body><p>Показання.</p>"
                 "<p>Симптоматичне лікування головного болю.</p></body></html>").encode("cp1251")
    header = b"<html><body><p>header</p></body></html>"
    data = _mht([("header.htm", header, "windows-1251"), ("main.htm", main_html, "windows-1251")])
    out = mht_text(data)
    assert "Симптоматичне лікування" in out and "header" not in out


def test_mht_without_any_charset_declaration_defaults_to_cp1251():
    """Catches: the default charset changing to utf-8 -- old MOZ .mht files declare no charset at all
    and would turn into mojibake."""
    body = "<html><body><p>Протипоказання. Виразкова хвороба.</p></body></html>".encode("cp1251")
    assert "Протипоказання" in mht_text(_mht([("main.htm", body, None)]))


def test_mht_without_html_part_is_empty():
    assert mht_text(_mht([])) == ""


def test_any_text_dispatches_on_file_signature():
    """Catches: a PDF treated as HTML (empty text -> leaflet silently skipped) or vice versa."""
    with mock.patch.object(doc_text, "pdf_text", return_value="from pdf") as pdf:
        assert any_text(b"%PDF-1.4 binary...") == "from pdf" and pdf.called
    assert "hello" in any_text(b"<html><body><p>hello</p></body></html>")
    mht = _mht([("main.htm", b"<html><body><p>mht body</p></body></html>", "utf-8")])
    assert "mht body" in any_text(mht, name="instr.mht")
    assert "mht body" in any_text(mht)                        # also recognised without the file name
    assert any_text(b"\x00\x01 unknown binary") == ""


def test_tidy_removes_file_path_headers_soft_hyphens_and_extra_blank_lines():
    """Catches: PDF page headers like "C:\\Users\\...\\ulotka.docx" and soft hyphens polluting chunks
    (they break keyword matching: "alko\u00adholu" would not match "alkohol")."""
    raw = "C:\\Users\\x\\ulotka.docx\nNie pić alko\u00adholu.\n\n\n\nDawka:\t400\u00a0mg  \n"
    out = _tidy(raw)
    assert out == "Nie pić alkoholu.\n\nDawka: 400 mg"


# ---------- the parsed sections feed the chunker ----------
@pytest.mark.parametrize("lang, file", [("pl", "pl_ibuprofen.txt"), ("cz", "cz_ibuprofen.txt"),
                                        ("es", "es_ibuprofeno.txt"), ("fr", "fr_ibuprofene.txt")])
def test_every_section_yields_at_least_one_non_empty_chunk(lang, file):
    """Catches: a section that parses fine but produces no chunks (e.g. whitespace-only text)."""
    for label, text in sections_pl(leaflet(file), lang):
        chunks = chunk_text(text)
        assert chunks and all(c.strip() for c in chunks), label
        assert all(len(c) <= 900 * 1.5 for c in chunks), label
