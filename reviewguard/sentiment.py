import hashlib, json, logging
from functools import lru_cache
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from .config import ROOT, SENTIMENT_REVISION

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def vader():
    return SentimentIntensityAnalyzer()


@lru_cache(maxsize=1)
def transformer():
    from transformers import pipeline
    import torch

    torch.set_num_threads(2)
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        model_kwargs={"cache_dir": str(ROOT / "models/hf_cache")},
        revision=SENTIMENT_REVISION,
        device=-1,
    )


@lru_cache(maxsize=1)
def transformer_status():
    try:
        return transformer(), None
    except Exception as exc:
        logger.warning("Sentiment fallback: %s", type(exc).__name__)
        return (
            None,
            f"Transformer unavailable ({type(exc).__name__}); VADER fallback active.",
        )


def analyze_sentiments(texts: list[str], backend: str = "transformer") -> list[dict]:
    cache = ROOT / "models/sentiment_cache"
    cache.mkdir(exist_ok=True)
    model, warning = transformer_status() if backend == "transformer" else (None, None)
    actual = "distilbert" if model is not None else "vader"
    results = [None] * len(texts)
    missing = []
    for i, text in enumerate(texts):
        key = hashlib.sha256((actual + text).encode()).hexdigest()
        path = cache / f"{key}.json"
        if path.exists():
            results[i] = json.loads(path.read_text())
        else:
            missing.append((i, text, path))
    if model is not None and missing:
        try:
            import torch

            with torch.no_grad():
                preds = model(
                    [t for _, t, _ in missing],
                    batch_size=16,
                    truncation=True,
                    max_length=256,
                )
        except Exception as exc:
            logger.warning("Transformer inference fallback: %s", type(exc).__name__)
            return analyze_sentiments(texts, "vader")
    else:
        preds = [None] * len(missing)
    for (i, text, path), pred in zip(missing, preds):
        c = (
            (1 if pred["label"] == "POSITIVE" else -1) * float(pred["score"])
            if pred
            else vader().polarity_scores(text)["compound"]
        )
        r = dict(
            compound=c,
            confidence=float(pred["score"]) if pred else abs(c),
            label="Positive" if c > 0.05 else "Negative" if c < -0.05 else "Neutral",
            backend=actual,
            warning=warning,
        )
        path.write_text(json.dumps(r))
        results[i] = r
    if warning:
        for r in results:
            r["warning"] = warning
    return results


def analyze_sentiment(text: str, backend: str = "transformer") -> dict:
    return analyze_sentiments([text], backend)[0]
