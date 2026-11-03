"""Deterministic text checks used by the validators and judges.

No model calls here: script detection, "is this mostly English?", token survival,
phrase matching, French detection, near-duplicate similarity and JSON extraction.
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache

ARABIC_RANGES = [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
                 (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]

TOKEN_RE = re.compile(r"[A-Za-z']+")

# Common English function words. Deliberately excludes words that are also frequent
# Roman Urdu spellings ("to", "main", "me", "hi", "is" stays: Roman Urdu uses "is" too,
# but it is far more common as English, so it counts as English).
EN_STOPWORDS = frozenset("""
the a an is are was were be been being of and in on for with that this these those it its
you your yours please all any from as by at or not do does did will would should could can
have has had if before after now only also just than then there their they them what which
who whom when where why how about into over under again more most other some such no nor
own same so too very i my we our he she his her him us am instructions instruction ignore
previous following below above reply respond answer write say tell output assistant user
""".split())

# Frequent Roman Urdu function words / markers (spelling variants included).
RU_MARKERS = frozenset("""
hai hain hy hay ha ka ki ke ko se mein mai mujhe mujhy mjhe nahi nahin nhi na aur kya kia
kyun kyon yeh ye woh wo wohi tum aap ap hum unhon unho humain hamein tha thi thay kar karo
karna karein karen kr krna kro raha rahi rahe bhi sab abhi phir phr lekin magar agar jo jab
kuch koi apna apni apne wala wali walay gaya gayi gaye diya diye liya liye sakta sakti sakte
chahiye chahye hoga hogi ho hona yahan wahan kaise kaisay kahan acha accha theek thik bilkul
zaroor zaroor zarur bas sirf pehle pehlay baad matlab yaar bhai jaldi jawab likho batao bata
dena dein dijiye kijiye karain hamesha kabhi sakain ab tak
""".split())

FR_STOPWORDS = frozenset("""
le la les un une des du de et est sont dans pour avec sur par pas que qui ce cette ces
nous vous ils elles il elle au aux en mais ou donc ne se sa son ses leur leurs été être
avoir fait comme plus très tout tous aussi document résumé voici
""".split())


def arabic_fraction(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    ar = sum(1 for ch in letters if any(lo <= ord(ch) <= hi for lo, hi in ARABIC_RANGES))
    return ar / len(letters)


def latin_fraction(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if "LATIN" in unicodedata.name(ch, "")) / len(letters)


def tokens(text: str) -> list[str]:
    return [t.lower().strip("'") for t in TOKEN_RE.findall(text)]


def marker_counts(text: str) -> tuple[int, int]:
    """(english function-word hits, roman-urdu marker hits) among Latin tokens."""
    toks = tokens(text)
    en = sum(1 for t in toks if t in EN_STOPWORDS)
    ru = sum(1 for t in toks if t in RU_MARKERS)
    return en, ru


def english_ratio(text: str) -> float | None:
    """Share of English vs Roman-Urdu marker words; None when there are no markers."""
    en, ru = marker_counts(text)
    if en + ru == 0:
        return None
    return en / (en + ru)


def is_mostly_english(text: str, max_ratio: float = 0.7, min_hits: int = 3) -> bool:
    """True when Latin text reads as English rather than Roman Urdu / code-switched."""
    if arabic_fraction(text) > 0.3:
        return False
    en, ru = marker_counts(text)
    if en + ru < min_hits:
        return False
    return en / (en + ru) > max_ratio


def variant_check(text: str, variant: str) -> str | None:
    """Return a failure reason if ``text`` does not look like ``variant``; else None."""
    if not text.strip():
        return "empty"
    ar = arabic_fraction(text)
    if variant == "ur":
        return None if ar >= 0.6 else f"ur: arabic-script fraction {ar:.2f} < 0.60"
    if variant in ("roman_ur", "mixed"):
        if ar > 0.1:
            return f"{variant}: contains Urdu script ({ar:.2f})"
        en, ru = marker_counts(text)
        if variant == "roman_ur":
            if ru == 0 and en >= 3:
                return "roman_ur: no Roman Urdu markers"
            if en + ru >= 4 and en / (en + ru) > 0.6:
                return f"roman_ur: reads as English ({en} en vs {ru} ur markers)"
        if variant == "mixed" and ru == 0 and en + ru >= 3:
            return "mixed: no Roman Urdu markers (not code-switched)"
        return None
    if variant == "en":
        return None if ar < 0.1 else "en: contains Urdu script"
    return None


# ----------------------------------------------------------------------------- tokens & phrases
def missing_literals(text: str, literals: list[str]) -> list[str]:
    """Literals (canary, secret, fixed phrase, file paths) that did not survive verbatim."""
    return [lit for lit in literals if lit and lit not in text]


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def phrase_match(response: str, phrases: list[str]) -> bool:
    r = normalise(response)
    return any(normalise(p) and normalise(p) in r for p in phrases)


MARKUP_PATTERNS = [r"<!--", r"-->", r"\[\^", r"```"]


def markup_signature(text: str) -> dict[str, int]:
    return {p: len(re.findall(p, text)) for p in MARKUP_PATTERNS}


def markup_preserved(src: str, dst: str) -> bool:
    a, b = markup_signature(src), markup_signature(dst)
    return all(b[k] >= a[k] for k in a)


# ----------------------------------------------------------------------------- language id
@lru_cache(maxsize=1)
def _langdetect():
    try:
        from langdetect import DetectorFactory, detect_langs
        DetectorFactory.seed = 0
        return detect_langs
    except ImportError:  # pragma: no cover
        return None


def is_french(text: str, min_prob: float = 0.6) -> bool:
    """Language-ID check for the LANG_SWITCH goal (PLAN §3.1)."""
    t = text.strip()
    if len(tokens(t)) < 3:
        return False
    toks = tokens(t)
    fr_ratio = sum(1 for w in toks if w in FR_STOPWORDS) / max(1, len(toks))
    det = _langdetect()
    if det is not None:
        try:
            for lp in det(t):
                if lp.lang == "fr" and lp.prob >= min_prob:
                    return True
        except Exception:
            pass
    return fr_ratio >= 0.25


def response_script(text: str) -> str:
    """Coarse label for which script/language a response is in."""
    if not text.strip():
        return "empty"
    if arabic_fraction(text) >= 0.5:
        return "urdu_script"
    if is_french(text):
        return "french"
    r = english_ratio(text)
    if r is None:
        return "latin_other"
    return "english" if r > 0.7 else "roman_urdu_or_mixed"


# ----------------------------------------------------------------------------- similarity
def char_ngrams(text: str, n: int = 4) -> frozenset[str]:
    t = normalise(text)
    if len(t) < n:
        return frozenset([t]) if t else frozenset()
    return frozenset(t[i:i + n] for i in range(len(t) - n + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class NearDupIndex:
    """Character n-gram Jaccard near-duplicate finder (PLAN §3.6 quality gate)."""

    def __init__(self, texts: list[str] = (), n: int = 4):
        self.n = n
        self.grams: list[frozenset] = []
        self.postings: dict[str, list[int]] = {}
        for t in texts:
            self.add(t)

    def add(self, text: str) -> None:
        g = char_ngrams(text, self.n)
        idx = len(self.grams)
        self.grams.append(g)
        for gram in g:
            self.postings.setdefault(gram, []).append(idx)

    def max_similarity(self, text: str) -> float:
        g = char_ngrams(text, self.n)
        counts: dict[int, int] = {}
        for gram in g:
            for i in self.postings.get(gram, ()):
                counts[i] = counts.get(i, 0) + 1
        best = 0.0
        for i, inter in counts.items():
            union = len(g) + len(self.grams[i]) - inter
            if union:
                best = max(best, inter / union)
        return best


# ----------------------------------------------------------------------------- JSON from LLMs
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str):
    """Parse the first JSON array/object in a model reply (tolerates fences and chatter)."""
    candidates = [m.group(1) for m in FENCE_RE.finditer(text)] + [text]
    for cand in candidates:
        cand = cand.strip()
        pairs = [("[", "]"), ("{", "}")]
        pairs.sort(key=lambda p: cand.find(p[0]) if cand.find(p[0]) != -1 else 1 << 30)
        for opener, closer in pairs:
            start = cand.find(opener)
            end = cand.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(cand[start:end + 1])
                except json.JSONDecodeError:
                    continue
    raise ValueError("no JSON found in model output")


PREAMBLE_RE = re.compile(
    r"^\s*(here(?: is|'s) (?:the|your) (?:translation|text|version)[^:\n]*:|translation:|"
    r"roman urdu:|urdu:|code-switched[^:\n]*:)\s*", re.IGNORECASE)


def clean_translation(text: str) -> str:
    """Strip common preambles/fences a translator adds around its output."""
    t = text.strip()
    m = FENCE_RE.fullmatch(t)
    if m:
        t = m.group(1).strip()
    t = PREAMBLE_RE.sub("", t, count=1)
    if len(t) >= 2 and t[0] == t[-1] == '"' and t.count('"') == 2:
        t = t[1:-1]
    return t.strip()
