"""Harvest-time emotion scoring.

Default path is lexicon-only (NRCLex + TextBlob) so GitHub Actions
and Streamlit Cloud stay alive. Optional GoEmotions batch path uses
sigmoid multi-label, not softmax-as-if-single-class.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

PRIMARY = ("joy", "anger", "fear", "sadness", "surprise")

NRC_TO_PRIMARY = {
    "joy": "joy",
    "positive": "joy",
    "trust": "joy",
    "anticipation": "surprise",
    "anger": "anger",
    "disgust": "anger",
    "fear": "fear",
    "sadness": "sadness",
    "negative": "sadness",
    "surprise": "surprise",
}

GO_COLLAPSE = {
    "joy": ["joy", "amusement", "excitement", "optimism", "love", "relief", "gratitude", "pride", "admiration", "approval", "caring"],
    "anger": ["anger", "annoyance", "disapproval", "disgust"],
    "fear": ["fear", "nervousness"],
    "sadness": ["sadness", "grief", "remorse", "disappointment", "embarrassment"],
    "surprise": ["surprise", "realization", "curiosity", "confusion"],
}

GO_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

_URL_RE = re.compile(r"https?://\S+|www\.\S+|reddit\.com/\S+", re.I)
_NON_LETTERS = re.compile(r"[^a-zA-Z\s']")


def clean_text(text: str) -> str:
    text = _URL_RE.sub(" ", str(text or ""))
    text = _NON_LETTERS.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def empty_primary() -> dict[str, float]:
    return {k: 0.0 for k in PRIMARY}


def nrc_primary(text: str) -> dict[str, float]:
    cleaned = clean_text(text)
    base = empty_primary()
    if len(cleaned.split()) < 3:
        return base
    try:
        from nrclex import NRCLex
    except Exception:
        return base
    scores = NRCLex(cleaned).raw_emotion_scores or {}
    for emotion, val in scores.items():
        bucket = NRC_TO_PRIMARY.get(emotion)
        if bucket:
            base[bucket] += float(val)
    return base


def textblob_polarity(text: str) -> float:
    cleaned = clean_text(text)
    if not cleaned:
        return 0.0
    try:
        from textblob import TextBlob
        return float(TextBlob(cleaned).sentiment.polarity)
    except Exception:
        return 0.0


def normalize(vec: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, v) for v in vec.values()) or 1.0
    return {k: max(0.0, v) / total for k, v in vec.items()}


def blend(a: dict[str, float], b: dict[str, float], w_b: float = 0.7) -> dict[str, float]:
    keys = set(a) | set(b)
    out = {k: (1.0 - w_b) * a.get(k, 0.0) + w_b * b.get(k, 0.0) for k in keys}
    return normalize(out)


def collapse_goemotions(label_scores: dict[str, float]) -> dict[str, float]:
    reduced = empty_primary()
    for bucket, labels in GO_COLLAPSE.items():
        reduced[bucket] = sum(float(label_scores.get(lab, 0.0)) for lab in labels)
    return normalize(reduced)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class GoEmotionsBatch:
    """Optional. Loads once. Scores a list of texts with sigmoid multi-label."""

    def __init__(self, model_dir: str | None = None):
        self.ok = False
        self.tokenizer = None
        self.model = None
        self.labels = list(GO_LABELS)
        self._load(model_dir)

    def _load(self, model_dir: str | None) -> None:
        try:
            from pathlib import Path
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            self.torch = torch
            candidates = []
            if model_dir:
                candidates.append(Path(model_dir))
            root = Path("models/goemotions_model")
            if root.exists():
                candidates.append(root)
                candidates.extend([p for p in root.iterdir() if p.is_dir()])
            folder = None
            for c in candidates:
                if (c / "config.json").exists():
                    folder = c
                    break
            if folder is None:
                # public multi-label checkpoint — only if caller wants network
                name = "SamLowe/roberta-base-go_emotions"
                self.tokenizer = AutoTokenizer.from_pretrained(name)
                self.model = AutoModelForSequenceClassification.from_pretrained(name)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(str(folder), local_files_only=True)
            self.model.eval()
            id2label = getattr(self.model.config, "id2label", None) or {}
            if id2label:
                self.labels = [id2label[i] for i in range(len(id2label))]
            self.ok = True
        except Exception:
            self.ok = False

    def score_many(self, texts: list[str], batch_size: int = 16) -> list[dict[str, float]]:
        if not self.ok:
            return [empty_primary() for _ in texts]
        out: list[dict[str, float]] = []
        torch = self.torch
        for i in range(0, len(texts), batch_size):
            chunk = [clean_text(t)[:400] or "neutral" for t in texts[i : i + batch_size]]
            enc = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            )
            with torch.no_grad():
                logits = self.model(**enc).logits
            # multi-label: sigmoid, not softmax
            probs = torch.sigmoid(logits).cpu().tolist()
            for row in probs:
                raw = {self.labels[j]: float(row[j]) for j in range(min(len(self.labels), len(row)))}
                out.append(collapse_goemotions(raw))
        return out


def score_records(records: list[dict], use_transformer: bool = False) -> list[dict]:
    texts = [
        " ".join(
            [
                str(r.get("title") or ""),
                str(r.get("selftext") or "")[:200],
            ]
        )
        for r in records
    ]
    nrc_vecs = [nrc_primary(t) for t in texts]
    go_vecs = [empty_primary() for _ in texts]
    if use_transformer:
        engine = GoEmotionsBatch()
        if engine.ok:
            go_vecs = engine.score_many(texts)

    scored = []
    for rec, nrc, go in zip(records, nrc_vecs, go_vecs):
        if use_transformer and any(go.values()):
            mood = blend(normalize(nrc), go, w_b=0.7)
            source = "nrc+goemotions"
        else:
            mood = normalize(nrc)
            source = "nrclex"
        scored.append(
            {
                **rec,
                "joy": mood["joy"],
                "anger": mood["anger"],
                "fear": mood["fear"],
                "sadness": mood["sadness"],
                "surprise": mood["surprise"],
                "polarity": textblob_polarity(" ".join([str(rec.get("title") or ""), str(rec.get("selftext") or "")[:200]])),
                "score_source": source,
            }
        )
    return scored


def aggregate(scored: Iterable[dict], key: str | None = None) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in scored:
        k = str(row.get(key) or "all") if key else "all"
        groups.setdefault(k, []).append(row)

    out: dict[str, dict] = {}
    for k, rows in groups.items():
        n = len(rows) or 1
        mood = empty_primary()
        polar = 0.0
        for r in rows:
            for e in PRIMARY:
                mood[e] += float(r.get(e) or 0.0)
            polar += float(r.get("polarity") or 0.0)
        mood = {e: mood[e] / n for e in PRIMARY}
        out[k] = {
            "n": len(rows),
            "mood": normalize(mood) if sum(mood.values()) else empty_primary(),
            "polarity": polar / n,
            "dominant": max(mood, key=mood.get) if any(mood.values()) else "neutral",
        }
    return out


def top_themes(titles: list[str], k: int = 12) -> list[tuple[str, int]]:
    stop = {
        "the", "and", "for", "that", "with", "this", "from", "have", "just",
        "about", "what", "when", "your", "will", "they", "them", "their",
        "not", "are", "was", "were", "been", "you", "but", "all", "can",
        "out", "how", "why", "who", "has", "had", "its", "into", "over",
        "after", "new", "now", "one", "like", "get", "got", "dont",
    }
    bag: Counter[str] = Counter()
    for title in titles:
        for tok in clean_text(title).lower().split():
            if len(tok) < 4 or tok in stop:
                continue
            bag[tok] += 1
    return bag.most_common(k)


def prophet_line(overall: dict, themes: list[tuple[str, int]], regional: dict) -> str:
    """Data-true one-liner. No toy GPT completion."""
    mood = overall.get("mood") or empty_primary()
    dom = max(mood, key=mood.get) if any(mood.values()) else "uneasy quiet"
    theme_bit = ", ".join(t for t, _ in themes[:4]) or "the usual noise"
    hottest = None
    hottest_score = -1.0
    for region, payload in regional.items():
        m = payload.get("mood") or {}
        peak = max(m.values()) if m else 0
        if peak > hottest_score:
            hottest_score = peak
            hottest = region
    region_bit = hottest.replace("_", " ") if hottest else "no one region"
    pct = round(100 * mood.get(dom, 0.0))
    return (
        f"The street is running {dom} ({pct}%) — themes: {theme_bit}. "
        f"Loudest desk: {region_bit}."
    )
