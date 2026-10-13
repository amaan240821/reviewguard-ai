import json
import numpy as np, pandas as pd, pytest
from reviewguard.config import Config, ROOT
from reviewguard.preprocess import clean_text, tokens, normalize, lemmatize, sentences
from reviewguard.data import load_dataset, split_data, validate_batch
from reviewguard.linguistic import linguistic_features, matched_spans
from reviewguard.scoring import score_signals, rating_mismatch, duplication_signal
from reviewguard.embeddings import embed_texts
from reviewguard.similarity import cosine_matrix, lexical_matrix, nearest_neighbors
from reviewguard.clustering import build_clusters
from reviewguard.pipeline import analyze_review, analyze_batch
from reviewguard.explain import highlight_phrases
from reviewguard.report import csv_bytes, pdf_bytes

OFFLINE = Config(sentiment_backend="vader", embedding_backend="lsa")


def test_clean_raw_separation():
    raw = "<p>I can't sleep!</p> https://hotel.test me@hotel.test"
    cleaned = clean_text(raw)
    assert (
        "<p>" not in cleaned
        and "cannot" in cleaned
        and "URL" in cleaned
        and "EMAIL" in cleaned
    )
    assert "!" in raw and raw != cleaned


def test_tokens_sentences_lemmas():
    assert tokens("The rooms", True) == ["rooms"]
    assert lemmatize("Rooms beds stayed") == ["room", "bed", "stay"]
    assert len(sentences("One night. Two rooms!")) == 2


def test_dataset_counts_and_splits():
    df = load_dataset()
    a, b = split_data(df)
    assert len(df) == 1596 and df.hotel.nunique() == 20
    assert len(a) == 1276 and len(b) == 320
    assert set(a.index).isdisjoint(b.index)
    assert set(a.text.map(normalize)).isdisjoint(b.text.map(normalize))
    assert b.groupby(["label", "polarity"]).size().between(79, 81).all()


@pytest.mark.parametrize(
    "data,mapping",
    [
        (pd.DataFrame({"text": []}), {"text": "text"}),
        (pd.DataFrame({"text": [""]}), {"text": "text"}),
        (pd.DataFrame({"text": [None]}), {"text": "text"}),
        (pd.DataFrame({"text": ["!!!"]}), {"text": "text"}),
        (pd.DataFrame({"text": ["Okay"]}), {"text": "missing"}),
        (
            pd.DataFrame({"text": ["Okay"], "rating": [6]}),
            {"text": "text", "rating": "rating"},
        ),
        (
            pd.DataFrame({"text": ["Okay"], "rating": ["five"]}),
            {"text": "text", "rating": "rating"},
        ),
    ],
)
def test_invalid_batches(data, mapping):
    with pytest.raises(ValueError):
        validate_batch(data, mapping)


def test_batch_limit():
    with pytest.raises(ValueError):
        validate_batch(pd.DataFrame({"text": ["Okay"] * 3}), {"text": "text"}, limit=2)


def test_missing_rating_allowed():
    df = validate_batch(
        pd.DataFrame({"review": ["Okay room"], "rating": [None]}),
        {"text": "review", "rating": "rating"},
    )
    assert pd.isna(df.rating.iloc[0])


@pytest.mark.parametrize(
    "value,band",
    [
        (0, "Low"),
        (30, "Low"),
        (31, "Moderate"),
        (60, "Moderate"),
        (61, "High"),
        (80, "High"),
        (81, "Very High"),
        (100, "Very High"),
    ],
)
def test_risk_boundaries(value, band):
    score, b, contrib = score_signals({k: value / 100 for k in "DCLRE"})
    assert score == pytest.approx(value) and b == band
    assert sum(contrib.values()) == pytest.approx(score)


def test_missing_weight_renormalization():
    signals = {"D": 1.0, "C": 0.0, "L": 0.0, "R": None, "E": 0.0}
    s, _, c = score_signals(signals)
    assert s == pytest.approx(100 * 0.3 / 0.85) and c["R"] == 0
    assert rating_mismatch(0.98, 1) == pytest.approx(0.99)
    assert rating_mismatch(1, None) is None


@pytest.mark.parametrize(
    "bad",
    [
        {"D": -1, "C": 1, "L": 0, "R": 0, "E": 1},
        {"D": 1, "C": 1, "L": 1, "R": 1, "E": 1},
    ],
)
def test_weight_validation(bad):
    with pytest.raises(ValueError):
        Config(weights=bad)


def test_language_features():
    f = linguistic_features("AMAZING AMAZING!!!!! BUY NOW!!!")
    assert f["uppercase_ratio"] > 0.9 and f["adjacent_repeats"] == 1
    assert (
        f["punctuation_ratio"] > 0 and f["promotional_hits"] >= 2 and f["score"] > 0.7
    )
    assert all(0 <= x <= 1 for x in f["subscores"].values())


def test_highlighting_safe():
    assert "&lt;script&gt;" in highlight_phrases("<script>amazing</script>")
    assert "<mark" in highlight_phrases("AMAZING!!!")
    assert len(matched_spans("amazing!!!")) >= 2


def test_similarity():
    texts = [
        "Clean room beside the train station",
        "Clean room beside the train station",
        "Volcanic rocks orbit Jupiter",
    ]
    values, _ = embed_texts(texts, "lsa")
    m = cosine_matrix(values)
    assert m[0, 1] == pytest.approx(1, abs=1e-5) and m[0, 2] < 0.2
    assert nearest_neighbors(m, 0)[0][0] == 1
    assert lexical_matrix(texts)[0, 1] == pytest.approx(1, abs=1e-5)


def test_graph_components():
    m = np.array([[1, 0.95, 0.1], [0.95, 1, 0.1], [0.1, 0.1, 1]])
    labels, groups, edges = build_clusters(m, 0.85)
    assert labels == [0, 0, -1] and groups[0]["size"] == 2 and len(edges) == 1


def test_signals_and_behavior():
    examples = json.loads((ROOT / "data/samples/single_examples.json").read_text())
    detailed = analyze_review(examples[0]["text"], config=OFFLINE)
    promotional = analyze_review(examples[1]["text"], config=OFFLINE)
    contradiction = analyze_review(examples[3]["text"], rating=1, config=OFFLINE)
    assert detailed.score <= 30
    assert promotional.score > 60
    assert contradiction.signals["R"] > 0.8
    assert detailed.signals["R"] is None and detailed.signals["D"] is None
    assert sum(detailed.contributions.values()) == pytest.approx(detailed.score)
    json.dumps(detailed.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "text,rating",
    [
        ("", None),
        ("!!!", None),
        ("Okay", 0),
        ("Okay", 6),
        ("Okay", float("nan")),
        ("a" * 20001, None),
    ],
)
def test_invalid_single(text, rating):
    with pytest.raises(ValueError):
        analyze_review(text, rating, config=OFFLINE)


def test_three_review_semantic_demo():
    df = pd.read_csv(ROOT / "data/samples/three_review_demo.csv")
    result = analyze_batch(df, {"text": "text", "rating": "rating"})
    assert len(result.clusters) == 1 and result.clusters[0]["size"] == 3


@pytest.fixture(scope="module")
def demo_result():
    return analyze_batch(
        pd.read_csv(ROOT / "data/samples/demo_batch.csv"),
        {"text": "text", "rating": "rating"},
    )


def test_injected_group_recovery(demo_result):
    df = demo_result.reviews
    for _, g in df[df.demo_injected].groupby("demo_group"):
        assert g.cluster_id.nunique() == 1 and g.cluster_id.iloc[0] >= 0
    assert df.suspicion_score.between(0, 100).all()


def test_exports(demo_result):
    df = pd.read_csv(__import__("io").BytesIO(csv_bytes(demo_result)))
    assert len(df) == 193 and set("DCLRE").issubset(df.columns)
    pdf = pdf_bytes(demo_result)
    import pymupdf

    doc = pymupdf.open(stream=pdf, filetype="pdf")
    assert len(doc) > 1 and "ReviewGuard" in doc[0].get_text()
    assert "screening" in doc[0].get_text().lower()


def test_batch_serialization(demo_result):
    json.dumps(demo_result.to_dict(), allow_nan=False)


def test_real_evaluation_artifacts():
    m = json.loads((ROOT / "reports/metrics.json").read_text())
    cm = np.array(m["test_classifier"]["confusion_matrix"])
    assert cm.sum() == 320
    assert (cm.trace() / cm.sum()) == pytest.approx(m["test_classifier"]["accuracy"])
    assert len(m["model_comparison"]) == 4 and len(m["hotel_folds"]) == 5
    assert len(list((ROOT / "reports/figures").glob("*.png"))) == 10


def test_missing_classifier_continues(monkeypatch):
    import reviewguard.classifier as classifier

    monkeypatch.setattr(classifier, "load_model", lambda: None)
    result = analyze_review(
        "The room had a comfortable bed and a clean bathroom.", config=OFFLINE
    )
    assert result.signals["C"] is None
    assert any("Classifier unavailable" in w for w in result.warnings)
    assert sum(result.effective_weights.values()) == pytest.approx(1)


def test_embedding_fallback(monkeypatch):
    import reviewguard.embeddings as embeddings

    def unavailable():
        raise OSError("test offline mode")

    monkeypatch.setattr(embeddings, "sentence_model", unavailable)
    vectors, backend = embeddings.embed_texts(
        ["unique fallback test bedroom view", "unrelated rainy airport afternoon"],
        "sentence-transformer",
    )
    assert backend == "tfidf-lsa" and vectors.shape[0] == 2


@pytest.mark.parametrize(
    "content,error",
    [(b"", "empty"), (b"text\n\xff", "UTF-8"), (b'text\n"unclosed', "parsed")],
)
def test_upload_errors(content, error):
    from app.ui.views import _read_upload

    class Upload:
        size = len(content)

        def getvalue(self):
            return content

    with pytest.raises(ValueError, match=error):
        _read_upload(Upload())
