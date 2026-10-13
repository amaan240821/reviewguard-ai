import json, re
import numpy as np
from .config import ROOT
from .preprocess import tokens, sentences

PHRASES = json.loads((ROOT / "data/lexicons/promotional.json").read_text())


def matched_spans(text):
    spans = []
    for phrase in PHRASES:
        spans.extend(
            dict(
                start=m.start(),
                end=m.end(),
                phrase=m.group(),
                reason="Promotional or extreme wording; context may be legitimate.",
            )
            for m in re.finditer(re.escape(phrase), text, re.I)
        )
    spans.extend(
        dict(
            start=m.start(),
            end=m.end(),
            phrase=m.group(),
            reason="Repeated emphatic punctuation.",
        )
        for m in re.finditer(r"[!?]{2,}", text)
    )
    return sorted(spans, key=lambda s: (s["start"], -s["end"]))


def linguistic_features(text):
    ts = tokens(text)
    n = max(len(ts), 1)
    words = re.findall(r"\b[A-Za-z]+\b", text)
    features = dict(
        word_count=len(ts),
        lexical_diversity=len(set(ts)) / n,
        uppercase_ratio=sum(w.isupper() and len(w) > 1 for w in words)
        / max(len(words), 1),
        punctuation_ratio=min((text.count("!") + text.count("?")) / n, 1),
        promotional_hits=sum(text.lower().count(p) for p in PHRASES),
        first_person_ratio=sum(t in {"i", "me", "my", "we", "our"} for t in ts) / n,
        detail_hits=sum(
            t
            in {
                "room",
                "bed",
                "bathroom",
                "floor",
                "breakfast",
                "desk",
                "shower",
                "elevator",
                "window",
                "train",
                "walk",
                "minutes",
            }
            for t in ts
        ),
        adjacent_repeats=sum(a == b for a, b in zip(ts, ts[1:])),
        average_sentence_length=n / max(len(sentences(text)), 1),
    )
    subs = dict(
        promotion=min(features["promotional_hits"] / 3, 1),
        punctuation=min(features["punctuation_ratio"] / 0.15, 1),
        capitals=min(features["uppercase_ratio"] / 0.35, 1),
        repetition=(
            min((1 - features["lexical_diversity"]) / 0.7, 1)
            if n > 6
            else min(features["adjacent_repeats"] / 2, 1)
        ),
        generic=float(features["detail_hits"] == 0 and not re.search(r"\d", text)),
        short=float(n < 15),
    )
    weights = dict(
        promotion=0.25,
        punctuation=0.20,
        capitals=0.20,
        repetition=0.10,
        generic=0.15,
        short=0.10,
    )
    features["subscores"] = subs
    features["score"] = float(sum(subs[k] * v for k, v in weights.items()))
    return features
