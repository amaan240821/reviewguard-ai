import json, html, io
import numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
from reviewguard.config import ROOT, Config
from reviewguard.data import load_dataset, split_data
from reviewguard.pipeline import analyze_review, analyze_batch
from reviewguard.report import csv_bytes, pdf_bytes
from reviewguard.scoring import NAMES
from reviewguard.clustering import build_clusters
from .theme import header, disclaimer, safe_error
from .components import advanced, result_panel
from .charts import styled, network, COLORS, PALETTE


@st.cache_data(show_spinner=False)
def dataset():
    return load_dataset()


@st.cache_data(show_spinner=False)
def cached_single(text, rating, reference, config):
    return analyze_review(text, rating, reference, config)


@st.cache_data(show_spinner=False)
def cached_batch(data, mapping, config):
    return analyze_batch(data, mapping, config)


def home():
    st.markdown(
        '<div class="eyebrow">REVIEW INTELLIGENCE / OVERVIEW</div><span class="status"><i></i> LOCAL WORKSPACE</span><span class="status">EXPLAINABLE NLP</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero"><div class="radar" aria-hidden="true"></div><div class="eyebrow">LOOK BEYOND THE STARS</div><h1>Every review has a story.<br><span>Find the hidden patterns.</span></h1><p>Screen suspicious language. Connect similar reviews. Turn five NLP signals into clear, inspectable evidence.</p></div>',
        unsafe_allow_html=True,
    )
    a, b, c = st.columns([1, 1, 1.2])
    if a.button("Analyze a review →", type="primary", width="stretch"):
        st.switch_page("pages/1_Single_Review.py")
    if b.button("Upload a batch ↗", width="stretch"):
        st.switch_page("pages/2_Batch_Forensics.py")
    if c.button("Try the 3-review demo", width="stretch"):
        try:
            with st.spinner("Connecting the three reviews…"):
                demo = pd.read_csv(ROOT / "data/samples/three_review_demo.csv")
                st.session_state["batch_result"] = cached_batch(
                    demo, {"text": "text", "rating": "rating"}, Config()
                )
                st.session_state["batch_name"] = "Three-review synthetic demonstration"
                st.session_state["batch_request"] = (
                    demo,
                    {"text": "text", "rating": "rating"},
                )
            st.switch_page("pages/2_Batch_Forensics.py")
        except Exception as exc:
            safe_error(exc)
    try:
        df = dataset()
        cols = st.columns(4)
        for col, title, value in zip(
            cols,
            [
                "Unique corpus reviews",
                "Hotel domains",
                "Corpus deceptive labels",
                "Independent signals",
            ],
            [f"{len(df):,}", df.hotel.nunique(), int(df.label.sum()), 5],
        ):
            col.metric(title, value)
        st.caption(
            "Live corpus counts after normalization and deduplication. Corpus labels describe the research dataset."
        )
    except FileNotFoundError:
        st.info(
            "Dataset not installed. Run make data to activate corpus exploration and training."
        )
    st.markdown(
        '<div class="flow"><span>01 · INPUT</span><b>→</b><span>02 · NLP</span><b>→</b><span>03 · SIGNALS</span><b>→</b><span>04 · SUSPICION</span><b>→</b><span>05 · EVIDENCE</span></div>',
        unsafe_allow_html=True,
    )
    st.subheader("A clearer view of review behaviour")
    cols = st.columns(3)
    cards = [
        (
            "01 / INSPECT",
            "Single-review intelligence",
            "Inspect five signals, highlighted phrases, rating contradictions, and evidence that supports a balanced reading.",
        ),
        (
            "02 / CONNECT",
            "Batch & cluster forensics",
            "Upload a CSV, map columns, find shared wording, and explore groups of closely related reviews.",
        ),
        (
            "03 / VERIFY",
            "Measured, transparent models",
            "Explore actual holdout metrics, hotel generalisation, ablations, and the tradeoffs behind the score.",
        ),
    ]
    for col, (icon, title, text) in zip(cols, cards):
        col.markdown(
            f'<div class="feature"><div class="icon">{icon}</div><h3>{title}</h3><p>{text}</p></div>',
            unsafe_allow_html=True,
        )
    disclaimer()


def single():
    header(
        "INSPECT / SINGLE REVIEW",
        "Read between the lines.",
        "Analyze language, sentiment, and similarity. Follow each signal back to its evidence.",
    )
    examples = json.loads((ROOT / "data/samples/single_examples.json").read_text())
    choice = st.selectbox(
        "Try an example",
        ["Write your own"] + [e["name"] for e in examples],
        key="example_choice",
    )
    if choice != "Write your own" and choice != st.session_state.get("loaded_example"):
        example = next(e for e in examples if e["name"] == choice)
        st.session_state["review_text"] = example["text"]
        st.session_state["star_rating"] = example["rating"]
        st.session_state["include_rating"] = True
        st.session_state["loaded_example"] = choice
    left, right = st.columns([2, 1], gap="large")
    with left:
        text = st.text_area(
            "Review text",
            height=210,
            placeholder="Paste a review to inspect its signals…",
            key="review_text",
            max_chars=20000,
        )
    with right:
        st.markdown("### Context")
        has_rating = st.checkbox("Include a star rating", key="include_rating")
        rating = st.slider(
            "Star rating", 1, 5, 4, key="star_rating", disabled=not has_rating
        )
        use_ref = st.checkbox("Compare with training hotel corpus", value=True)
        st.caption(
            "Training reviews and prior session analyses form the reference pool. Corpus similarity is domain-specific."
        )
    config = advanced("single")
    refs = []
    if use_ref:
        try:
            refs = split_data(dataset())[0].text.tolist()
        except FileNotFoundError:
            st.info("No corpus available. Similarity will use session reviews only.")
        refs += st.session_state.get("session_reviews", [])
        refs = refs[: config.max_batch_size]
    clicked = st.button("Analyze review →", type="primary", width="stretch")
    if clicked:
        try:
            with st.spinner("Encoding review and gathering evidence…"):
                st.session_state["single_request"] = (
                    text,
                    rating if has_rating else None,
                    refs,
                )
                st.session_state["single_result"] = cached_single(
                    text, rating if has_rating else None, refs, config
                )
                history = st.session_state.get("session_reviews", [])
                if text not in history:
                    st.session_state["session_reviews"] = history + [text]
        except Exception as exc:
            safe_error(exc)
            st.session_state.pop("single_request", None)
            st.session_state.pop("single_result", None)
    elif "single_request" in st.session_state:
        try:
            st.session_state["single_result"] = cached_single(
                *st.session_state["single_request"], config
            )
        except Exception as exc:
            safe_error(exc)
    if "single_result" in st.session_state:
        st.divider()
        result_panel(st.session_state["single_result"])
    else:
        st.info(
            "Add a review or choose an example, then run the analysis. Your evidence will appear here."
        )
    disclaimer()


def _read_upload(upload):
    if upload.size > 10 * 1024 * 1024:
        raise ValueError("File exceeds 10 MB. Use a smaller CSV.")
    try:
        return pd.read_csv(io.BytesIO(upload.getvalue()))
    except UnicodeDecodeError:
        raise ValueError(
            "CSV must use UTF-8 encoding. Re-export the file as UTF-8 CSV."
        )
    except pd.errors.EmptyDataError:
        raise ValueError("CSV is empty. Include a header and at least one review.")
    except pd.errors.ParserError:
        raise ValueError(
            "CSV could not be parsed. Check separators and quotation marks."
        )


def batch():
    header(
        "CONNECT / BATCH FORENSICS",
        "Find the patterns at scale.",
        "From a CSV to an evidence workspace: ranked reviews, shared wording, and similarity clusters.",
    )
    a, b = st.columns([2, 1])
    with a:
        upload = st.file_uploader(
            "Upload review CSV · UTF-8 · up to 10 MB / 5,000 rows", type=["csv"]
        )
    with b:
        st.markdown("### Explore a prepared case")
        load_demo = st.button("Load full demo batch", width="stretch")
        st.caption(
            "180 real corpus reviews + 13 clearly flagged synthetic examples. Corpus has no star ratings."
        )
        st.download_button(
            "↓ Demo CSV",
            (ROOT / "data/samples/demo_batch.csv").read_bytes(),
            file_name="demo_batch.csv",
            mime="text/csv",
        )
    data = None
    name = None
    if upload:
        try:
            data = _read_upload(upload)
            name = upload.name
        except Exception as exc:
            safe_error(exc)
    if load_demo:
        data = pd.read_csv(ROOT / "data/samples/demo_batch.csv")
        name = "Demo corpus + synthetic injections"
        st.session_state["pending_batch"] = data
        st.session_state["pending_name"] = name
    elif data is None and "pending_batch" in st.session_state:
        data = st.session_state["pending_batch"]
        name = st.session_state["pending_name"]
    config = advanced("batch")
    if data is not None:
        with st.expander("Map columns & preview", expanded=True):
            cols = st.columns(3)
            options = list(data.columns)
            default = next(
                (
                    c
                    for c in options
                    if c.lower() in ["text", "review", "review_text", "content"]
                ),
                options[0] if options else None,
            )
            if not options:
                st.error("CSV has no columns.")
                return
            textcol = cols[0].selectbox(
                "Review text *", options, index=options.index(default)
            )
            optional = ["Not provided"] + options
            ratingcol = cols[1].selectbox(
                "Star rating",
                optional,
                index=optional.index("rating") if "rating" in optional else 0,
            )
            hotelcol = cols[2].selectbox(
                "Hotel / product",
                optional,
                index=optional.index("hotel") if "hotel" in optional else 0,
            )
            d, e = st.columns(2)
            reviewer = d.selectbox("Reviewer ID", optional)
            date = e.selectbox("Date", optional)
            st.caption(
                "Optional metadata is preserved in exports; account and temporal signals are not inferred."
            )
            st.dataframe(data.head(5), hide_index=True, width="stretch")
        mapping = {
            "text": textcol,
            "rating": None if ratingcol == "Not provided" else ratingcol,
            "hotel": hotelcol,
            "reviewer": reviewer,
            "date": date,
        }
        if (
            st.button("Run batch forensics →", type="primary", width="stretch")
            or load_demo
        ):
            try:
                bar = st.progress(0, text="1 / 3 · Validate and encode")
                with st.spinner("Computing five signals and similarity communities…"):
                    result = cached_batch(data, mapping, config)
                    bar.progress(0.85, text="2 / 3 · Signals and clustering complete")
                    st.session_state["batch_result"] = result
                    st.session_state["batch_name"] = name
                    st.session_state["batch_request"] = (data, mapping)
                bar.progress(1.0, text="3 / 3 · Evidence ready")
            except Exception as exc:
                safe_error(exc)
        elif "batch_request" in st.session_state:
            old_data, old_map = st.session_state["batch_request"]
            try:
                st.session_state["batch_result"] = cached_batch(
                    old_data, old_map, config
                )
            except Exception as exc:
                safe_error(exc)
    if "batch_result" in st.session_state:
        batch_dashboard(st.session_state["batch_result"])
    else:
        st.info(
            "Upload a CSV or load the demo. Results stay in this session for the Cluster Explorer."
        )
    disclaimer()


def batch_dashboard(result):
    df = result.reviews
    st.divider()
    st.caption(st.session_state.get("batch_name", "Current batch"))
    cols = st.columns(5)
    values = [
        len(df),
        f"{df.suspicion_score.mean():.1f}",
        f'{df.risk_band.isin(["High","Very High"]).mean()*100:.1f}%',
        len(result.clusters),
        len(result.edges),
    ]
    for col, label, value in zip(
        cols,
        [
            "Reviews",
            "Mean suspicion",
            "High / very high",
            "Similarity clusters",
            "Near-duplicate pairs",
        ],
        values,
    ):
        col.metric(label, value)
    st.caption(
        f'Embeddings: {result.embedding_backend} · Sentiment: {result.results[0].sentiment["backend"]} · Signals use design weights. Synthetic rows are demonstration-only.'
    )
    if "MiniLM" not in result.embedding_backend:
        st.info(
            "TF-IDF / LSA fallback is active. Similarity values may differ from MiniLM."
        )
    tabs = st.tabs(["Overview", "Review evidence", "Coordinated patterns", "Export"])
    with tabs[0]:
        a, b = st.columns(2)
        a.plotly_chart(
            styled(
                px.histogram(
                    df,
                    x="suspicion_score",
                    nbins=20,
                    labels={"suspicion_score": "Suspicion score / 100"},
                )
            ),
            width="stretch",
        )
        b.plotly_chart(
            styled(
                px.pie(
                    df,
                    names="risk_band",
                    hole=0.65,
                    color="risk_band",
                    color_discrete_map=COLORS,
                )
            ),
            width="stretch",
        )
        a, b = st.columns(2)
        a.plotly_chart(
            styled(px.histogram(df, x="sentiment", color="sentiment")), width="stretch"
        )
        average = pd.DataFrame(
            {
                "Signal": [NAMES[k] for k in "DCLRE"],
                "Mean contribution": [
                    np.mean([r.contributions[k] for r in result.results])
                    for k in "DCLRE"
                ],
            }
        )
        b.plotly_chart(
            styled(px.bar(average, x="Mean contribution", y="Signal", orientation="h")),
            width="stretch",
        )
        a, b = st.columns(2)
        length = df.assign(
            word_count=[r.features["word_count"] for r in result.results]
        )
        a.plotly_chart(
            styled(
                px.scatter(
                    length,
                    x="word_count",
                    y="suspicion_score",
                    color="risk_band",
                    color_discrete_map=COLORS,
                    labels={"word_count": "Review length (words)"},
                )
            ),
            width="stretch",
        )
        if df.rating.notna().any():
            b.plotly_chart(
                styled(
                    px.scatter(
                        df.dropna(subset=["rating"]),
                        x="rating",
                        y="suspicion_score",
                        color="risk_band",
                        color_discrete_map=COLORS,
                    )
                ),
                width="stretch",
            )
        else:
            b.info(
                "No ratings supplied. Rating mismatch is omitted and weights renormalized."
            )
    with tabs[1]:
        a, b = st.columns([2, 1])
        query = a.text_input("Search review text")
        bands = b.multiselect("Risk bands", list(COLORS), default=list(COLORS))
        filtered = df[
            df.risk_band.isin(bands)
            & df.text.str.contains(query, case=False, regex=False)
        ].sort_values("suspicion_score", ascending=False)
        show = ["text", "suspicion_score", "risk_band", "sentiment", "cluster_id"] + (
            ["demo_injected"] if "demo_injected" in df else []
        )
        st.dataframe(
            filtered[show],
            hide_index=False,
            width="stretch",
            column_config={
                "suspicion_score": st.column_config.ProgressColumn(
                    "Suspicion / 100", min_value=0, max_value=100, format="%.1f"
                )
            },
        )
        if len(filtered):
            index = st.selectbox(
                "Inspect review",
                filtered.index.tolist(),
                format_func=lambda i: f"#{i+1} · {df.suspicion_score.iloc[i]:.1f}/100 · {str(df.text.iloc[i])[:70]}",
            )
            result_panel(result.results[index], key="batch_detail")
        else:
            st.info("No reviews match these filters.")
    with tabs[2]:
        st.caption(
            "Groups are connected components of a similarity graph. An indirect path can connect reviews that are not themselves near-duplicates."
        )
        if not result.clusters:
            st.info("No similarity cluster meets the current threshold.")
        if result.pair_evidence:
            with st.expander("Near-duplicate pair evidence · semantic and lexical"):
                st.dataframe(
                    pd.DataFrame(result.pair_evidence), hide_index=True, width="stretch"
                )
        for cluster in result.clusters:
            with st.expander(
                f'Cluster {cluster["cluster_id"]+1} · {cluster["size"]} reviews · coordination indicator {cluster["coordination_score"]:.1f}'
            ):
                st.write(
                    f'Mean pairwise similarity: {cluster["mean_similarity"]:.3f} · Mean suspicion: {cluster["mean_suspicion"]:.1f}'
                )
                st.write("Shared terms: " + ", ".join(cluster["shared_terms"]))
                st.dataframe(
                    df.iloc[cluster["members"]][show], hide_index=False, width="stretch"
                )
        if st.button("Open Cluster Explorer →"):
            st.switch_page("pages/3_Cluster_Explorer.py")
    with tabs[3]:
        st.subheader("Take the evidence with you")
        a, b = st.columns(2)
        a.download_button(
            "↓ Annotated CSV",
            csv_bytes(result),
            file_name="reviewguard-forensics.csv",
            mime="text/csv",
        )
        b.download_button(
            "↓ Forensics PDF",
            pdf_bytes(result),
            file_name="reviewguard-forensics.pdf",
            mime="application/pdf",
        )
        st.download_button(
            "↓ Similarity pairs CSV",
            pd.DataFrame(result.pair_evidence).to_csv(index=False),
            file_name="reviewguard-pairs.csv",
            mime="text/csv",
        )
        st.caption(
            "CSV preserves original columns and includes each signal, score, cluster ID, and explanation. PDF contains a summary and ranked evidence."
        )


def clusters():
    header(
        "EXPLORE / SIMILARITY GRAPH",
        "Connect the evidence.",
        "Explore shared wording and semantic neighbourhoods. Move the threshold to see which connections persist.",
    )
    if "batch_result" not in st.session_state:
        st.info("Run a batch analysis first, or start with the demo.")
        if st.button("Load demo for graph", type="primary"):
            try:
                with st.spinner("Building the similarity graph…"):
                    st.session_state["batch_result"] = cached_batch(
                        pd.read_csv(ROOT / "data/samples/demo_batch.csv"),
                        {"text": "text", "rating": "rating"},
                        Config(),
                    )
            except Exception as exc:
                safe_error(exc)
        if "batch_result" not in st.session_state:
            disclaimer()
            return
    r = st.session_state["batch_result"]
    threshold = st.slider("Graph similarity threshold", 0.6, 0.99, 0.85, 0.01)
    labels, stats, edges = build_clusters(r.similarity, threshold)
    a, b = st.columns([2.1, 1])
    with a:
        st.plotly_chart(network(r, threshold), width="stretch")
        st.caption(
            "Node size = suspicion. Colour = graph component. Grey = isolated review. Hover for review text."
        )
        if len(r.reviews) > 800:
            st.warning(
                "Graph displays the first 800 reviews for responsiveness; cluster statistics use the full batch."
            )
    with b:
        st.metric("Connected groups", len(stats))
        st.metric("Connections", len(edges))
        if stats:
            cid = st.selectbox(
                "Select a cluster",
                [s["cluster_id"] for s in stats],
                format_func=lambda x: f"Cluster {x+1}",
            )
            group = next(s for s in stats if s["cluster_id"] == cid)
            st.write(
                f'{group["size"]} reviews · mean similarity {group["mean_similarity"]:.3f}'
            )
            for i in group["members"][:15]:
                with st.expander(
                    f"Review {i+1} · {r.reviews.suspicion_score.iloc[i]:.1f}"
                ):
                    st.write(r.reviews.text.iloc[i])
            if group["size"] > 15:
                st.caption(
                    "First 15 examples shown; complete group remains in batch exports."
                )
        else:
            st.info("No connected groups at this threshold.")
    disclaimer()


def evaluation():
    header(
        "VERIFY / MODEL EVALUATION",
        "Evidence behind the engine.",
        "Real runs. Frozen thresholds. Transparent tradeoffs. Positive class: deceptive corpus label.",
    )
    path = ROOT / "reports/metrics.json"
    if not path.exists():
        st.info("No evaluation has been computed. Run make train and make evaluate.")
        disclaimer()
        return
    m = json.loads(path.read_text())
    which = st.radio(
        "Evaluation protocol",
        ["Holdout classifier", "Holdout composite", "Hotel-held-out classifier"],
        horizontal=True,
    )
    key = {
        "Holdout classifier": "test_classifier",
        "Holdout composite": "test_composite",
        "Hotel-held-out classifier": "hotel_held_out",
    }[which]
    vals = m[key]
    cols = st.columns(6)
    for col, label, k in zip(
        cols,
        ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
        ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"],
    ):
        col.metric(label, f"{vals[k]:.3f}")
    st.caption(
        f'Computed {m["timestamp"]} · {m["dataset"]["train_rows"]} training / {m["dataset"]["test_rows"]} holdout reviews · decision threshold {vals["threshold"]:.2f}'
    )
    st.info(m["hotel_protocol"] + " " + m["rating_signal"])
    tabs = st.tabs(
        [
            "Performance",
            "Model comparison",
            "Ablations & weights",
            "Cluster validation",
            "Artifacts",
        ]
    )
    with tabs[0]:
        a, b = st.columns(2)
        cm = np.array(vals["confusion_matrix"])
        a.plotly_chart(
            styled(
                px.imshow(
                    cm,
                    x=["Truthful", "Deceptive"],
                    y=["Truthful", "Deceptive"],
                    text_auto=True,
                    color_continuous_scale="Blues",
                    labels={"x": "Predicted", "y": "Actual", "color": "Count"},
                )
            ),
            width="stretch",
        )
        b.image(str(ROOT / "reports/figures/07_roc_curve.png"))
        a, b = st.columns(2)
        a.image(str(ROOT / "reports/figures/08_precision_recall.png"))
        b.image(str(ROOT / "reports/figures/03_scores_by_label.png"))
        st.caption(m["threshold_selection"])
    with tabs[1]:
        comparison = pd.DataFrame(m["model_comparison"]).drop(columns="fold_scores")
        st.dataframe(comparison, hide_index=True, width="stretch")
        st.plotly_chart(
            styled(
                px.bar(
                    comparison, x="cv_f1", y="model", orientation="h", error_x="cv_std"
                )
            ),
            width="stretch",
        )
        st.caption(
            "Five-fold CV on training data only. If MiniLM is unavailable, the embedding candidates use fold-fitted LSA and the backend is reported."
        )
    with tabs[2]:
        st.dataframe(
            pd.DataFrame(m["ablations"]).drop(columns="confusion_matrix"),
            hide_index=True,
            width="stretch",
        )
        st.subheader("Design weights vs learned coefficients")
        st.json(m["weights"])
        st.caption(
            "Learned logistic coefficients are signed and are not normalized design weights. Both protocols use training-only threshold selection."
        )
    with tabs[3]:
        r = m["synthetic_cluster_recovery"]
        st.metric("Injected-group pair recall", f'{r["pair_recall"]:.1%}')
        st.write(
            f'{r["recovered_pairs"]} of {r["total_pairs"]} injected within-group pairs recovered.'
        )
        st.warning(r["description"])
    with tabs[4]:
        st.download_button(
            "↓ Metrics JSON",
            path.read_bytes(),
            file_name="metrics.json",
            mime="application/json",
        )
        for p in sorted((ROOT / "reports/figures").glob("*.png")):
            with st.expander(p.stem.replace("_", " ")):
                st.image(str(p))
                st.caption("Generated from the current evaluation run.")
    disclaimer()


def dataset_explorer():
    header(
        "DISCOVER / DATASET",
        "Know the data.",
        "Explore the corpus behind the classifier: class labels, sentiment polarity, hotel coverage, and vocabulary.",
    )
    try:
        df = dataset()
    except FileNotFoundError:
        st.info("Dataset missing. Run make data.")
        disclaimer()
        return
    cols = st.columns(4)
    for c, l, v in zip(
        cols,
        ["Unique reviews", "Truthful", "Deceptive", "Hotels"],
        [len(df), int((df.label == 0).sum()), int(df.label.sum()), df.hotel.nunique()],
    ):
        c.metric(l, v)
    st.caption(
        "Original download: 1,600 rows. Four duplicated texts removed. No star ratings in the corpus."
    )
    a, b = st.columns(2)
    a.plotly_chart(
        styled(px.histogram(df, x="deceptive", color="polarity", barmode="group")),
        width="stretch",
    )
    b.plotly_chart(
        styled(px.histogram(df, x="hotel", color="deceptive", barmode="stack")),
        width="stretch",
    )
    a, b = st.columns(2)
    a.plotly_chart(
        styled(
            px.histogram(
                df.assign(words=df.text.str.split().str.len()),
                x="words",
                color="deceptive",
                nbins=30,
                barmode="overlay",
                opacity=0.6,
            )
        ),
        width="stretch",
    )
    with b:
        label = st.selectbox("Top terms for label", ["truthful", "deceptive"])
        from sklearn.feature_extraction.text import CountVectorizer

        vec = CountVectorizer(stop_words="english", max_features=3000)
        values = vec.fit_transform(df[df.deceptive == label].text)
        counts = np.asarray(values.sum(axis=0)).ravel()
        ids = np.argsort(counts)[-15:]
        terms = pd.DataFrame(
            {"term": vec.get_feature_names_out()[ids], "count": counts[ids]}
        )
        st.plotly_chart(
            styled(px.bar(terms, x="count", y="term", orientation="h")), width="stretch"
        )
    query = st.text_input("Search the corpus")
    hotel = st.selectbox(
        "Hotel filter", ["All hotels"] + sorted(df.hotel.unique().tolist())
    )
    view = df[df.text.str.contains(query, case=False, regex=False)]
    if hotel != "All hotels":
        view = view[view.hotel == hotel]
    st.dataframe(view.drop(columns="label"), hide_index=True, width="stretch")
    st.caption(f"{len(view):,} matching reviews")
    st.markdown(
        "Source: [Ott Deceptive Opinion Spam Corpus v1.4](https://myleott.com/op-spam.html). The original licence is included in data/raw/CORPUS_LICENSE.txt."
    )
    disclaimer()


def about():
    header(
        "UNDERSTAND / METHODOLOGY",
        "Useful signals. Human judgment.",
        "ReviewGuard is a research screening tool for explainable review analysis. It does not establish authenticity.",
    )
    st.markdown("### How the score works")
    st.latex(r"S=100\sum_{i\in A} \frac{w_i}{\sum_{j\in A}w_j}s_i")
    st.caption(
        "A is the set of available signals. Missing ratings are omitted, then weights are renormalized."
    )
    st.dataframe(
        pd.DataFrame(
            [
                dict(signal=k, name=NAMES[k], design_weight=f"{v:.0%}")
                for k, v in Config().weights.items()
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.info(
        "Design weights are design choices, not empirical constants. The evaluation compares them with a learned logistic signal model."
    )
    for title, text in [
        (
            "D · Semantic duplication",
            "MiniLM cosine similarity and character 5-gram similarity provide copy evidence. Similarities below 0.60 are ignored; evidence rises toward 0.95 and neighbour count adds a small contribution. Connected components use a default 0.85 edge threshold.",
        ),
        (
            "C · Deception classifier",
            "Training-only five-fold model comparison selects a linear text classifier. Sigmoid calibration happens inside training folds. This is a probability of the research corpus label, not a probability of real-world fraud.",
        ),
        (
            "L · Language anomaly",
            "A weighted mean of promotional phrases, punctuation, capitalization, repeated vocabulary, generic wording, and short length. These cues can also appear in legitimate reviews.",
        ),
        (
            "R · Rating mismatch",
            "A star rating maps linearly from 1–5 to expected polarity -1 to +1. Half the distance from observed polarity gives the mismatch. No rating means this signal is unavailable.",
        ),
        (
            "E · Extreme sentiment",
            "Only absolute sentiment above 0.80 contributes. It rises linearly to 1.0 and is downweighted for texts longer than 30 words.",
        ),
        (
            "Risk bands",
            "0–30 Low; >30–60 Moderate; >60–80 High; >80–100 Very High. Bands are operational design choices; the binary evaluation threshold is separately selected on training CV.",
        ),
    ]:
        with st.expander(title):
            st.write(text)
    st.markdown("### Limitations & responsible use")
    st.write(
        "Genuine reviews can sound promotional. Independent customers can express similar opinions. Sentiment may misread sarcasm or context. Similarity does not prove coordination. Performance varies across domains and languages; this model was evaluated on English hotel reviews. The corpus uses crowd-written deceptive examples and does not represent modern marketplace fraud or LLM-generated text."
    )
    st.write(
        "Injected groups validate a demonstration workflow only. Do not remove reviews or penalize authors solely on a score. Uploaded reviews are processed by the local Python process; model assets may download from Hugging Face on first use."
    )
    st.markdown("### Future scope")
    st.write(
        "Account behaviour, temporal bursts, multilingual data, product-specific models, reviewer–product graphs, and evaluation on naturally coordinated reviews require additional data and validation."
    )
    disclaimer()
