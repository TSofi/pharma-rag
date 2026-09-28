"""Turn official leaflets into clean text split into sections.

  PL  PDF patient leaflet (URPL) or EMA product information  -> sections 1-6 of the EU leaflet template
  UA  .mht (Word web archive) instruction from the MOZ register -> the fixed headings of Ukrainian instructions
"""
import email
import html
import re
from html.parser import HTMLParser


# ------------------------------------------------------------------ raw text
def pdf_text(data: bytes) -> str:
    import pymupdf  # pip install pymupdf (only needed for ingestion, not by the web server)

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        text = "\n".join(page.get_text() for page in doc)
    text = re.sub(r"(?m)^\s*(\d{1,2}\.)\s*\n\s*", r"\1 ", text)   # "1.\nCo to jest" -> "1. Co to jest"
    text = re.sub(r"(?m)^\s*([•\-–])\s*\n\s*", r"\1 ", text)       # lone bullets
    return _tidy(text)


class _HTMLText(HTMLParser):
    BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "td"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("style", "script", "head", "xml"):
            self.skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n" if tag != "td" else " | ")

    def handle_endtag(self, tag):
        if tag in ("style", "script", "head", "xml") and self.skip:
            self.skip -= 1
        elif tag in ("p", "div", "tr", "li") or tag.startswith("h") and len(tag) == 2:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data.replace("\r", " ").replace("\n", " "))


def html_text(markup: str) -> str:
    markup = re.sub(r"<!\[if[^>]*>.*?<!\[endif\]>|<!--.*?-->", " ", markup, flags=re.S)
    p = _HTMLText()
    p.feed(markup)
    return _tidy(html.unescape("".join(p.parts)))


def mht_text(data: bytes) -> str:
    msg = email.message_from_bytes(data)
    parts = [p for p in msg.walk() if p.get_content_type() == "text/html"]
    if not parts:
        return ""
    main = max(parts, key=lambda p: len(p.get_payload(decode=True) or b""))  # skip header.htm etc.
    raw = main.get_payload(decode=True) or b""
    charset = main.get_content_charset() or "cp1251"
    m = re.search(rb'charset=["\']?([\w-]+)', raw[:3000])
    if m and m.group(1).lower() not in (b"us-ascii",):
        charset = m.group(1).decode()
    return html_text(raw.decode(charset, errors="replace"))


def any_text(data: bytes, name: str = "") -> str:
    if data[:4] == b"%PDF":
        return pdf_text(data)
    if name.endswith(".mht") or b"multipart/related" in data[:2000].lower() or data[:5] == b"MIME-":
        return mht_text(data)
    if b"<html" in data[:3000].lower():
        return html_text(data.decode("utf-8", errors="replace"))
    return ""


def _tidy(text: str) -> str:
    text = re.sub(r"(?m)^[A-Z]:\\.*$", "", text)  # file-path page headers some PDFs carry
    text = text.replace("\xa0", " ").replace("­", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ------------------------------------------------------------------ sections
# EU patient-leaflet template (QRD), the same in every EU language. Canonical English names are used
# as section labels so the UI and prompts look the same across markets.
PL_SECTIONS = [
    (1, r"Co to jest", "What it is and what it is used for"),
    (2, r"Informacje ważne", "Before you take it (contraindications, warnings, interactions, pregnancy)"),
    (3, r"Jak (przyjmować|stosować|używać|podawać|jest podawany|się stosuje)", "How to take it (dosage)"),
    (4, r"Możliwe działania niepożądane", "Possible side effects"),
    (5, r"Jak przechowywać", "How to store it"),
    (6, r"Zawartość opakowania", "Contents of the pack and other information"),
]


# EMA product information PDFs: "B. <leaflet heading>" then "<leaflet title>" (one per strength).
EMA_LEAFLET = {
    "pl": (r"B\.\s*ULOTKA (DLA PACJENTA|INFORMACYJNA)", r"Ulotka dołączona do opakowania"),
    "cz": (r"B\.\s*PŘÍBALOVÁ INFORMACE", r"Příbalová informace\s*[:\-–]"),
    "es": (r"B\.\s*PROSPECTO", r"Prospecto\s*[:\-–]"),
    "fr": (r"B\.\s*NOTICE", r"Notice\s*[:\-–]"),
}


def leaflet_part(text: str, lang: str = "pl") -> str:
    """EMA product information = SmPC + labelling + leaflet; keep only the (first) patient leaflet."""
    head, title = EMA_LEAFLET.get(lang, EMA_LEAFLET["pl"])
    m = re.search(rf"(?m)^\s*{head}\s*$", text)
    if m:
        rest = text[m.end():]
        start = re.search(rf"(?im)^\s*{title}", rest)
        text = rest[start.start():] if start else rest
        nxt = re.search(rf"(?im)^\s*{title}", text[200:])  # several strengths -> keep the first one
        if nxt:
            text = text[:200 + nxt.start()]
    return text


# The same six QRD headings in other EU languages (label taken from PL_SECTIONS by number).
QRD = {
    "pl": [p for _, p, _ in PL_SECTIONS],
    "cz": [r"Co je", r"Čemu musíte věnovat pozornost", r"Jak se .{0,40}(užívá|používá|podává|aplikuje)|Jak .{0,20}(užívat|používat)",
           r"Možné nežádoucí účinky", r"Jak .{0,40}uchovávat", r"Obsah balení"],
    "es": [r"Qué es", r"Qué necesita saber", r"Cómo (tomar|usar|utilizar|se)", r"Posibles efectos adversos",
           r"Conservación", r"Contenido del envase"],
    "fr": [r"Qu.est.ce que", r"Quelles sont les informations", r"Comment (prendre|utiliser|est)",
           r"Quels sont les effets indésirables", r"Comment conserver", r"Contenu de l.emballage"],
}


def fr_notice(text: str) -> str:
    """A BDPM product page holds the SmPC first and the patient notice second, plus site navigation.
    Keep only the notice: from the last "ANSM - Mis à jour le" up to the page footer."""
    starts = [m.start() for m in re.finditer(r"ANSM - Mis à jour le", text)]
    if not starts:
        return ""
    text = text[starts[-1]:]
    end = re.search(r"Lien de redirection vers la plateforme|Mentions légales Lien|Plan du site Lien", text)
    text = text[:end.start()] if end else text
    return re.sub(r"\s*Redirection vers le haut de page", "", text).strip()


def sections_pl(text: str, lang: str = "pl") -> list[tuple[str, str]]:
    """EU patient leaflets (QRD template): split at the six numbered headings, in any supported language."""
    if lang == "fr" and "Base de Données Publique des Médicaments" in text:  # a BDPM page, not an EMA PDF
        text = fr_notice(text)
    else:
        text = leaflet_part(text, lang)
    if not text:
        return []
    heads = []
    for (num, _, label), pat in zip(PL_SECTIONS, QRD[lang]):
        found = list(re.finditer(rf"(?m)^\s*{num}\.?\s*({pat})", text, flags=re.I))
        if found:
            heads.append((found[-1].start() if len(found) > 1 else found[0].start(), label))  # last = after TOC
    heads.sort()
    if len(heads) < 3:
        return [("Leaflet", text)] if text.strip() else []
    out = [("Introduction", text[:heads[0][0]])] if heads[0][0] > 200 else []
    for i, (pos, label) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        out.append((label, text[pos:end]))
    return [(l, t.strip()) for l, t in out if len(t.strip()) > 40]


UA_SECTIONS = [
    (r"Склад", "Composition"),
    (r"Лікарська форма", "Dosage form"),
    (r"Фармакотерапевтична група", "Pharmacotherapeutic group"),
    (r"Фармакологічні властивості", "Pharmacological properties"),
    (r"Клінічні характеристики", None),
    (r"Показання", "Indications"),
    (r"Протипоказання", "Contraindications"),
    (r"Взаємодія з іншими лікарськими засобами", "Interactions"),
    (r"Особливості застосування", "Warnings and precautions"),
    (r"Застосування (у|в) період вагітності", "Pregnancy and breastfeeding"),
    (r"Здатність впливати на швидкість реакції", "Driving and using machines"),
    (r"Спосіб застосування та дози", "Dosage and administration"),
    (r"Діти", "Children"),
    (r"Передозування", "Overdose"),
    (r"Побічні реакції", "Side effects"),
    (r"Термін придатності", "Shelf life"),
    (r"Умови зберігання", "Storage"),
    (r"Упаковка", "Package"),
    (r"Категорія відпуску", "Prescription status"),
    (r"Виробник", "Manufacturer"),
]


def sections_ua(text: str) -> list[tuple[str, str]]:
    heads, pos = [], 0
    for pat, label in UA_SECTIONS:  # headings always come in this order, so search after the previous one
        m = re.compile(rf"(?m)^\s*{pat}[^\n]{{0,80}}?(\.|:|\n)", flags=re.I).search(text, pos)
        if m:
            heads.append((m.start(), label))
            pos = m.end()
    heads.sort()
    if len(heads) < 4:
        return [("Instruction", text)] if text.strip() else []
    out = []
    for i, (pos, label) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        if label and label != "Manufacturer":
            out.append((label, text[pos:end].strip()))
    return [(l, t) for l, t in out if len(t) > 30]
