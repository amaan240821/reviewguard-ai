"""Reproducible holdout evaluation and training-only signal-weight experiments."""

import json, logging
from datetime import datetime, timezone
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from .config import ROOT, Config
from .data import load_dataset, split_data
from .classifier import load_model, candidates
from .features import EmbeddingFeatures
from .sentiment import analyze_sentiments
from .linguistic import linguistic_features
from .scoring import duplication_signal

log = logging.getLogger(__name__)
SIGNALS = ["D", "C", "L", "E"]


def metrics(y, p, threshold=0.5):
    pred = np.asarray(p) >= threshold
    return dict(
        accuracy=float(accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y, p)),
        pr_auc=float(average_precision_score(y, p)),
        threshold=float(threshold),
        confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    )


def best_threshold(y, p):
    values = np.linspace(0.01, 0.99, 99)
    return float(
        max(values, key=lambda t: f1_score(y, np.asarray(p) >= t, zero_division=0))
    )


def signals_for(texts, probs, reftexts, same=False):
    enc = EmbeddingFeatures().fit(reftexts)
    ref = enc.transform(reftexts)
    vals = ref if same else enc.transform(texts)
    sem = np.clip(cosine_similarity(vals, ref), 0, 1)
    lex = TfidfVectorizer(analyzer="char", ngram_range=(5, 5)).fit(reftexts)
    lm = cosine_similarity(lex.transform(texts), lex.transform(reftexts))
    if same:
        np.fill_diagonal(sem, 0)
        np.fill_diagonal(lm, 0)
    d = np.array(
        [
            duplication_signal(
                float(s.max()), float(l.max()), int((np.maximum(s, l) >= 0.85).sum())
            )
            for s, l in zip(sem, lm)
        ]
    )
    sentiments = analyze_sentiments(list(texts))
    f = [linguistic_features(t) for t in texts]
    l = np.array([x["score"] for x in f])
    e = np.array(
        [
            np.clip((abs(s["compound"]) - 0.8) / 0.2, 0, 1)
            * min(1, 30 / max(x["word_count"], 1))
            for s, x in zip(sentiments, f)
        ]
    )
    return np.column_stack([d, probs, l, e]), sentiments, vals, enc.backend_


def evaluate_all():
    df = load_dataset()
    train, test = split_data(df)
    model = load_model()
    if model is None:
        raise RuntimeError("Run make train first.")
    card = json.loads((ROOT / "models/model_card.json").read_text())
    oof = np.load(ROOT / "models/train_oof.npz")
    lookup = dict(zip(oof["indices"], oof["probability"]))
    ptrain = np.array([lookup[i] for i in train.index])
    ptest = model.predict_proba(test.text)[:, 1]
    Xtr, _, _, backend = signals_for(train.text, ptrain, train.text, True)
    Xte, sentiments, embeddings, _ = signals_for(test.text, ptest, train.text)
    weights = np.array([Config().weights[k] for k in SIGNALS])
    weights /= weights.sum()
    tr = Xtr @ weights
    te = Xte @ weights
    threshold = best_threshold(train.label, tr)
    clf_threshold = best_threshold(train.label, ptrain)
    weight_model = LogisticRegression(C=1, random_state=42, max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    learned_oof = cross_val_predict(
        weight_model, Xtr, train.label, cv=cv, method="predict_proba"
    )[:, 1]
    learned_threshold = best_threshold(train.label, learned_oof)
    weight_model.fit(Xtr, train.label)
    learned = weight_model.predict_proba(Xte)[:, 1]
    rows = []
    for name, p, t in [
        ("Classifier only", ptest, clf_threshold),
        ("Composite design weights", te, threshold),
        ("Learned logistic weights", learned, learned_threshold),
    ]:
        rows.append(dict(experiment=name, **metrics(test.label, p, t)))
    for k in SIGNALS:
        w = weights.copy()
        w[SIGNALS.index(k)] = 0
        w /= w.sum()
        rows.append(
            dict(
                experiment="Without " + k,
                **metrics(test.label, Xte @ w, best_threshold(train.label, Xtr @ w)),
            )
        )
    rows.append(
        dict(
            experiment="Without R (absent in corpus)",
            **metrics(test.label, te, threshold),
        )
    )
    # Hotel-generalisation folds restricted to training pool to keep final test untouched.
    gh = np.zeros(len(train))
    group_rows = []
    for fold, (a, b) in enumerate(
        GroupKFold(n_splits=5).split(train.text, train.label, train.hotel)
    ):
        m = CalibratedClassifierCV(
            candidates()[card["algorithm"]], cv=3, method="sigmoid"
        )
        m.fit(train.text.iloc[a], train.label.iloc[a])
        gh[b] = m.predict_proba(train.text.iloc[b])[:, 1]
        group_rows.append(
            dict(
                fold=fold + 1,
                test_hotels=sorted(train.hotel.iloc[b].unique().tolist()),
                **metrics(train.label.iloc[b], gh[b], clf_threshold),
            )
        )
    pd.DataFrame(rows).to_csv(ROOT / "reports/tables/ablations.csv", index=False)
    pd.DataFrame(group_rows).to_csv(
        ROOT / "reports/tables/hotel_folds.csv", index=False
    )
    predictions = test.copy()
    predictions["classifier_probability"] = ptest
    predictions["suspicion_score"] = 100 * te
    for j, k in enumerate(SIGNALS):
        predictions[k] = Xte[:, j]
    predictions.to_csv(ROOT / "reports/tables/holdout_predictions.csv", index=False)
    learned_coefs = dict(zip(SIGNALS, weight_model.coef_[0].tolist()))
    output = dict(
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset=dict(
            raw_rows=1600,
            clean_rows=len(df),
            removed_duplicates=1600 - len(df),
            train_rows=len(train),
            test_rows=len(test),
            class_counts={str(k): int(v) for k, v in df.label.value_counts().items()},
        ),
        positive_class="deceptive",
        test_classifier=metrics(test.label, ptest, clf_threshold),
        test_composite=metrics(test.label, te, threshold),
        hotel_held_out=metrics(train.label, gh, clf_threshold),
        hotel_folds=group_rows,
        model_comparison=card["model_comparison"],
        ablations=rows,
        weights=dict(
            design={k: float(v) for k, v in zip(SIGNALS, weights)},
            learned_logistic_coefficients=learned_coefs,
            learned_intercept=float(weight_model.intercept_[0]),
        ),
        embedding_backend=backend,
        sentiment_backend=sentiments[0]["backend"],
        rating_signal="Unavailable: original corpus has no star ratings; R omitted and weights renormalized.",
        threshold_selection="Maximum F1 on out-of-fold training predictions; frozen before holdout evaluation. Learned-weight threshold uses a second training-only CV.",
        hotel_protocol="5-fold GroupKFold by hotel on the training pool only; final 20% holdout untouched.",
        limitations=[
            "Corpus deception labels do not measure real-world fraud.",
            "R cannot be validated on the unrated corpus.",
            "No natural coordination labels exist; cluster recovery measured on synthetic injections only.",
        ],
    )
    figures(output, df, test, Xte, te, ptest, sentiments, embeddings)
    # Measured synthetic demonstration recovery, separate from corpus classification.
    from .pipeline import analyze_batch

    demo = pd.read_csv(ROOT / "data/samples/demo_batch.csv")
    batch = analyze_batch(demo, {"text": "text", "rating": "rating"})
    recovered = 0
    total = 0
    for _, group in demo[demo.demo_injected].groupby("demo_group"):
        ids = group.index.tolist()
        for ai, a in enumerate(ids):
            for b in ids[ai + 1 :]:
                total += 1
                recovered += int(
                    batch.reviews.cluster_id.iloc[a] >= 0
                    and batch.reviews.cluster_id.iloc[a]
                    == batch.reviews.cluster_id.iloc[b]
                )
    output["synthetic_cluster_recovery"] = dict(
        recovered_pairs=recovered,
        total_pairs=total,
        pair_recall=recovered / total,
        description="Connected-component pair recall on injected groups; not real-world coordination accuracy.",
    )
    (ROOT / "reports/metrics.json").write_text(json.dumps(output, indent=2))
    write_captions(output)
    log.info("Test classifier: %s", output["test_classifier"])
    return output


def figures(output, df, test, X, score, probs, sentiments, embeddings):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    folder = ROOT / "reports/figures"
    folder.mkdir(exist_ok=True)

    def save(name):
        plt.tight_layout()
        plt.savefig(folder / (name + ".png"), dpi=300, bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(7, 4))
    df.deceptive.value_counts().plot.bar(color=["#4263eb", "#12b886"])
    plt.ylabel("Reviews")
    plt.xlabel("Corpus label")
    save("01_dataset_distribution")
    plt.figure(figsize=(7, 4))
    plt.hist(score * 100, bins=20, color="#4263eb")
    plt.xlabel("Suspicion score / 100")
    plt.ylabel("Heldout reviews")
    save("02_suspicion_distribution")
    plt.figure(figsize=(7, 4))
    for label, name, color in [(0, "Truthful", "#12b886"), (1, "Deceptive", "#4263eb")]:
        plt.hist(
            score[test.label.to_numpy() == label] * 100,
            bins=20,
            alpha=0.55,
            label=name,
            color=color,
        )
    plt.xlabel("Suspicion score / 100")
    plt.ylabel("Heldout reviews")
    plt.legend()
    save("03_scores_by_label")
    plt.figure(figsize=(5, 4))
    cm = np.array(output["test_classifier"]["confusion_matrix"])
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0, 1], ["Truthful", "Deceptive"])
    plt.yticks([0, 1], ["Truthful", "Deceptive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    save("04_confusion_matrix")
    plt.figure(figsize=(7, 4))
    pd.Series([s["label"] for s in sentiments]).value_counts().plot.bar(color="#12b886")
    plt.xlabel("Predicted sentiment")
    plt.ylabel("Heldout reviews")
    save("05_sentiment_distribution")
    from .embeddings import embed_texts
    from .similarity import cosine_matrix
    import networkx as nx

    demo = pd.read_csv(ROOT / "data/samples/three_review_demo.csv")
    v, _ = embed_texts(demo.text, "sentence-transformer")
    sim = cosine_matrix(v)
    G = nx.Graph()
    G.add_nodes_from(range(len(demo)))
    G.add_edges_from(
        (i, j)
        for i in range(len(demo))
        for j in range(i + 1, len(demo))
        if sim[i, j] >= 0.85
    )
    plt.figure(figsize=(6, 4))
    nx.draw_networkx(
        G, pos=nx.spring_layout(G, seed=42), node_color="#8b5cf6", font_color="white"
    )
    plt.title("Synthetic three-review demonstration")
    plt.axis("off")
    save("06_similarity_network")
    plt.figure(figsize=(6, 5))
    for p, name in [(probs, "Classifier"), (score, "Composite")]:
        a, b, _ = roc_curve(test.label, p)
        plt.plot(a, b, label=name)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    save("07_roc_curve")
    plt.figure(figsize=(6, 5))
    for p, name in [(probs, "Classifier"), (score, "Composite")]:
        precision, recall, _ = precision_recall_curve(test.label, p)
        plt.plot(recall, precision, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    save("08_precision_recall")
    plt.figure(figsize=(7, 4))
    plt.bar(SIGNALS, X.mean(axis=0), color=["#4263eb", "#8b5cf6", "#12b886", "#f59f00"])
    plt.ylabel("Mean normalized signal")
    plt.xlabel("Signal (R absent)")
    save("09_signal_importance")
    projection = TSNE(
        n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto"
    ).fit_transform(embeddings)
    plt.figure(figsize=(7, 5))
    plt.scatter(
        projection[:, 0], projection[:, 1], c=test.label, cmap="cool", alpha=0.7, s=14
    )
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.title("Heldout review embeddings; colour = corpus label")
    save("10_embedding_projection")


def write_captions(output):
    cm = output["test_classifier"]["confusion_matrix"]
    m = output["test_classifier"]
    captions = {
        "01_dataset_distribution": "After removing four duplicated texts, the corpus contains 1,596 unique reviews: 800 deceptive and 796 truthful.",
        "02_suspicion_distribution": "Design-weight suspicion scores on the untouched 320-review holdout. Scores are screening indicators, not calibrated fraud probabilities.",
        "03_scores_by_label": "Score distributions overlap; composite signals may reduce binary classification performance because they measure additional phenomena.",
        "04_confusion_matrix": f"Classifier at the frozen training-CV threshold: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}.",
        "05_sentiment_distribution": f'Predicted sentiment on the holdout using {output["sentiment_backend"]}. Sentiment labels are not deception labels.',
        "06_similarity_network": "Synthetic motivating example only. Similarity edges are evidence of shared wording, not proof of coordinated authorship.",
        "07_roc_curve": f'Holdout classifier ROC-AUC {m["roc_auc"]:.4f}; composite ROC-AUC {output["test_composite"]["roc_auc"]:.4f}.',
        "08_precision_recall": f'Classifier average precision {m["pr_auc"]:.4f}; positive class is deceptive.',
        "09_signal_importance": "Mean raw signal strength on the holdout, not causal importance. Rating is absent in this corpus.",
        "10_embedding_projection": "Seeded t-SNE projection of holdout embeddings. Distances in this visualization are exploratory, not a classifier.",
    }
    text = "# Report assets\n\nAll figures generated by `make evaluate` at 300 DPI.\n\n"
    for name, caption in captions.items():
        text += f"## {name}\n\n`reports/figures/{name}.png`\n\n{caption}\n\n"
    (ROOT / "docs/REPORT_ASSETS.md").write_text(text)
