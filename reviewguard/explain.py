import html, re, difflib
import numpy as np
from .linguistic import matched_spans
from .scoring import NAMES


def explanations(signals, rating_available):
    reasons = []
    if signals["C"] is not None and signals["C"] >= 0.6:
        reasons.append(
            "The classifier finds language associated with deceptive examples in its training corpus."
        )
    if signals["D"] is not None and signals["D"] >= 0.5:
        reasons.append(
            "Close semantic or lexical neighbours contribute to the duplication signal."
        )
    if signals["L"] >= 0.35:
        reasons.append(
            "Promotional language, punctuation, repetition, or limited detail contributes to the language signal."
        )
    if signals["R"] is not None and signals["R"] >= 0.5:
        reasons.append("The star rating and the expressed sentiment disagree.")
    if signals["E"] >= 0.8:
        reasons.append(
            "Very strong sentiment contributes a small amount; strong opinions alone do not establish deception."
        )
    if not reasons:
        reasons.append(
            "No strong individual signal was found. This does not verify authenticity."
        )
    if not rating_available:
        reasons.append(
            "Rating signal not used: no rating provided; available weights renormalized."
        )
    return reasons


def counter_evidence(features, text):
    out = []
    if features["detail_hits"] >= 3:
        out.append("Concrete details about the stay provide useful context.")
    if re.search(r"\d", text):
        out.append("Numbers or specific measurements add detail.")
    if re.search(r"\b(but|however|although|except)\b", text, re.I):
        out.append("The review includes a balanced or qualified statement.")
    return out or [
        "No strong counter-indicator found; absence of detail does not establish deception."
    ]


def highlight_phrases(text):
    parts = []
    start = 0
    for span in matched_spans(text):
        if span["start"] < start:
            continue
        parts.extend(
            [
                html.escape(text[start : span["start"]]),
                '<mark title="'
                + html.escape(span["reason"], quote=True)
                + '">'
                + html.escape(text[span["start"] : span["end"]])
                + "</mark>",
            ]
        )
        start = span["end"]
    parts.append(html.escape(text[start:]))
    return "".join(parts)


def word_diff(a, b):
    return " ".join(
        ("+" if t.startswith("+ ") else "-" if t.startswith("- ") else "") + t[2:]
        for t in difflib.ndiff(a.split(), b.split())
        if not t.startswith("? ")
    )


def influential_terms(text, limit=8):
    from .classifier import load_model

    model = load_model()
    if model is None:
        return []
    first = model.calibrated_classifiers_[0].estimator
    # Explain text coefficients only, including when the selected model has a union.
    if "features" in first.named_steps:
        tf = dict(first.named_steps["features"].transformer_list)["tfidf"]
        coef = first.named_steps["clf"].coef_[0][: len(tf.get_feature_names_out())]
    elif "tfidf" in first.named_steps:
        tf = first.named_steps["tfidf"]
        coef = first.named_steps["clf"].coef_[0]
    else:
        return []
    row = tf.transform([text])
    ids = row.indices
    v = row.data * coef[ids]
    return [
        {"term": str(tf.get_feature_names_out()[ids[i]]), "contribution": float(v[i])}
        for i in np.argsort(-np.abs(v))[:limit]
    ]
