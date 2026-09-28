"""Product index for the Polish and Ukrainian markets (built by scripts/fetch_markets.py).

Every product of the national register is known by name, even though only one leaflet per group
(same active substance + same form) is stored in the vector database. This module finds which products
a question mentions and which leaflet groups to search.
"""
import json
import re
from functools import lru_cache

from . import config

MARKET_FILE = config.ROOT / "data" / "markets" / "{}.json"

# Everyday words that also happen to start a product name; never treat them as a drug mention.
STOP = set("""alkohol alkoholem tabletki tabletka syrop krople witamina witaminy vitaminum maść woda aqua test
plus forte junior dzieci dziecko ciąża ciąży mleko kawa herbata ból bólu głowy gorączka przeziębienie lekarz
zdrowie żelazo magnez wapń cynk jod sól ranigast
алкоголь таблетки таблетка сироп краплі мазь вітамін вітаміни вода розчин діти дитина дітям вагітність
кава чай біль болить голова голови температура застуда лікар здоров'я залізо магній кальцій цинк йод сіль
water vitamin tablet tablets syrup drops pain alcohol coffee
children childrens children's infant infants adult adults extra maximum regular original daily night cold
allergy sleep headache cough sinus nasal stomach acid antacid relief sore throat skin hand body face baby
kids women natural organic medicated advanced ultra super triple total complete clear fresh cool arthritis
muscle back heart sunscreen broad spectrum first with take taking what does should while after before dose
much many when there their this that from have been will your about
mohu můžu užívat těhotenství dávka nežádoucí účinky bolest hlavy léky lék dítě děti alkohol
puedo tomar embarazo dosis efectos adversos dolor cabeza niños alcohol medicamento pastilla
peux prendre grossesse effets indésirables douleur enfants alcool médicament comprimé""".split())
MAX_GROUPS_PER_NAME = 30  # store brands ("Equate", "CVS Health") label hundreds of products: not a drug name

# US / English names -> names used in the European registers.
EN_ALIASES = {"acetaminophen": "paracetamol", "tylenol": "paracetamol", "albuterol": "salbutamol",
              "aspirin": "acetylsalicylic", "advil": "ibuprofen", "motrin": "ibuprofen"}


def norm(s: str) -> str:
    # є/е are mixed up in registered names ("НУРОФЄН" vs how people type "нурофен").
    return re.sub(r"[®™©*]", "", s.lower()).replace("є", "е").replace("’", "'").strip()


def stem(k: str) -> str:
    """Tolerate inflected endings: Polopiryna -> Polopiryną, Но-шпа -> Но-шпу, Voltaren -> Voltarenu."""
    return k if len(k) < 6 else k[:-1] if len(k) < 8 else k[:-2]


def pretty(name: str) -> str:
    w = re.split(r"\s+", name.replace("®", "").strip())[0].strip(",;")
    return w.capitalize() if w.isupper() and len(w) > 3 else w


def first_word(name: str) -> str:
    return re.split(r"[\s,;/()]+", norm(name))[0].strip("-.")


@lru_cache(maxsize=4)
def index(market: str) -> dict:
    path = MARKET_FILE.parent / f"{market}.json"
    if not path.exists():
        return {"keys": {}, "prefix": {}, "display": {}, "library": []}
    data = json.loads(path.read_text("utf-8"))
    keys: dict[str, set[str]] = {}
    display: dict[str, tuple] = {}  # key -> (display name, leaflet url of the shortest-named product, rx, full name)
    for name, gid, url, rx in data["products"]:
        k = first_word(name)
        if len(k) < 4 or k in STOP or k.isdigit():
            continue
        keys.setdefault(k, set()).add(gid)
        if k not in display or len(name) < len(display[k][3]):
            display[k] = (pretty(name), url, rx, name, gid)  # gid of the "main" product (shortest name)
    for gid, g in data["groups"].items():  # also find groups by English substance name ("ibuprofen")
        for n in (g.get("en") or "", g.get("inn") or ""):
            k = first_word(n)
            if len(k) >= 5 and k not in STOP:
                keys.setdefault(k, set()).add(gid)
                if k not in display or (not display[k][1] and len(n) < len(display[k][3])):
                    display[k] = (pretty(n), "", False, n, None)
    generic_keys = {first_word(g.get(f) or "") for g in data["groups"].values() for f in ("en", "inn")}
    for k in [k for k, v in keys.items() if len(v) > MAX_GROUPS_PER_NAME and k not in generic_keys]:
        del keys[k]
    prefix: dict[str, list[str]] = {}
    for k in keys:
        if len(k) >= 5:
            prefix.setdefault(k[:5], []).append(k)
    # Keep only what's needed at runtime (the raw product list is large; the server has 512 MB of RAM).
    lib = _library(data)
    keys = {k: sorted(v)[:25] for k, v in keys.items()}
    return {"keys": keys, "prefix": prefix, "display": display, "library": lib}


def detect(market: str, question: str) -> list[dict]:
    """Products/substances mentioned in the question. Handles Polish/Ukrainian inflection
    ("Nurofenu", "нурофеном") by allowing up to 3 extra letters after a known name."""
    idx = index(market)
    found: dict[str, dict] = {}
    for typed in re.findall(r"[\w'-]{4,}", norm(question)):
        word = EN_ALIASES.get(typed, typed)
        key = word if word in idx["keys"] else None
        if not key and len(word) >= 5:
            key = next((k for k in sorted(idx["prefix"].get(word[:5], []), key=len, reverse=True)
                        if word.startswith(stem(k)) and abs(len(word) - len(k)) <= 3), None)
        if key and key not in found:
            name, url, rx, _, main = idx["display"][key]
            if word != typed:  # "Tylenol" in the PL market: keep the user's word
                name = typed.capitalize()
            found[key] = {"asked": name, "key": key, "groups": idx["keys"][key], "url": url, "rx": rx,
                          "main": main}
    return list(found.values())


def _library(data: dict) -> list[dict]:
    """One entry per leaflet group, with the product names that share it (for the UI drawer)."""
    names: dict[str, list[str]] = {}
    for name, gid, _url, _rx in data["products"]:
        short = re.sub(r"\s+", " ", name.replace("®", "")).strip()
        lst = names.setdefault(gid, [])
        if short not in lst and len(lst) < 8:
            lst.append(short)
    out = []
    marked = any("indexed" in g for g in data["groups"].values())
    for gid, g in data["groups"].items():
        if not (g.get("indexed") if marked else g.get("cands")):
            continue
        title = (g.get("en") or g.get("inn") or names.get(gid, ["?"])[0]).lower()
        out.append({"drug": title, "brands": names.get(gid, []), "form": g["form"], "group": gid})
    return sorted(out, key=lambda d: (d["drug"], d["form"]))


def library(market: str) -> list[dict]:
    return index(market)["library"]
