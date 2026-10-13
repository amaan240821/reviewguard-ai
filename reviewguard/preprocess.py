import re, html, unicodedata
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def clean_text(text: str) -> str:
    """Clean model view; original text remains available for raw features."""
    text = unicodedata.normalize("NFKC", html.unescape(str(text)))
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"https?://\S+", " URL ", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b", " EMAIL ", text)
    for a, b in {
        "can't": "cannot",
        "won't": "will not",
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "I'm": "I am",
    }.items():
        text = text.replace(a, b)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str, remove_stopwords: bool = False) -> list[str]:
    ts = re.findall(r"[a-z]+(?:'[a-z]+)?", clean_text(text).lower())
    return [t for t in ts if t not in ENGLISH_STOP_WORDS] if remove_stopwords else ts


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?]+", clean_text(text)) if s.strip()]


def normalize(text: str) -> str:
    return " ".join(tokens(text))


def lemmatize(text: str) -> list[str]:
    """Conservative resource-free lemma map; no hidden NLP download."""
    exceptions = {
        "rooms": "room",
        "hotels": "hotel",
        "beds": "bed",
        "guests": "guest",
        "stayed": "stay",
    }
    return [exceptions.get(t, t) for t in tokens(text)]
