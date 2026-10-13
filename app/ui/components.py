import json, html
import pandas as pd
import streamlit as st
from reviewguard.config import ROOT, Config
from reviewguard.scoring import NAMES
from reviewguard.explain import highlight_phrases, word_diff
from reviewguard.report import pdf_bytes
from .charts import gauge, contributions, styled
import plotly.express as px


def advanced(key):
    with st.expander("Advanced settings · design weights and models"):
        st.caption(
            "These are configurable design choices, not scientific constants. Missing signals are omitted and remaining weights are renormalized."
        )
        cols = st.columns(5)
        weights = {}
        for col, k in zip(cols, "DCLRE"):
            weights[k] = col.slider(
                NAMES[k], 0, 100, int(Config().weights[k] * 100), key=f"{key}_{k}"
            )
        total = sum(weights.values())
        if not total:
            st.warning("At least one weight must be positive. Using defaults.")
            weights = Config().weights
        else:
            weights = {k: v / total for k, v in weights.items()}
        threshold = st.slider(
            "Similarity edge threshold", 0.6, 0.99, 0.85, 0.01, key=f"{key}_threshold"
        )
        fallback = st.checkbox(
            "Use lightweight offline models (VADER + TF-IDF / LSA)",
            key=f"{key}_fallback",
        )
    return Config(
        weights=weights,
        duplicate_threshold=threshold,
        sentiment_backend="vader" if fallback else "transformer",
        embedding_backend="lsa" if fallback else "sentence-transformer",
    )


def result_panel(r, key="single"):
    left, right = st.columns([0.85, 1.6], gap="large")
    with left:
        st.plotly_chart(gauge(r.score, r.band), width="stretch", key=key + "_gauge")
        st.caption(
            f'Sentiment: {r.sentiment["label"]} · {r.sentiment["backend"]} | Embeddings: {r.embedding_backend}'
        )
    with right:
        st.subheader(f"{r.band} suspicion · {r.score:.1f}/100")
        ordered = sorted(r.contributions, key=r.contributions.get, reverse=True)[:2]
        st.write(
            "Main contributors: " + ", ".join(NAMES[k].lower() for k in ordered) + "."
        )
        for reason in r.reasons:
            st.markdown("• " + reason)
        st.caption(
            "Rating signal not used: no rating provided."
            if r.signals["R"] is None
            else f'Rating mismatch signal: {r.signals["R"]:.2f}'
        )
    for warning in r.warnings:
        st.info(warning)
    tabs = st.tabs(
        [
            "Evidence & explanation",
            "Signal breakdown",
            "Similar reviews",
            "Model terms",
            "Raw & cleaned",
        ]
    )
    with tabs[0]:
        st.markdown(
            '<div class="review-text">' + highlight_phrases(r.text) + "</div>",
            unsafe_allow_html=True,
        )
        st.caption("Highlighted wording is an indicator; hover for context.")
        st.subheader("Counter-evidence")
        for item in r.counter_evidence:
            st.write("✓ " + item)
    with tabs[1]:
        st.plotly_chart(contributions(r), width="stretch", key=key + "_waterfall")
        st.dataframe(
            pd.DataFrame(
                [
                    dict(
                        signal=NAMES[k],
                        value=r.signals[k],
                        contribution_points=r.contributions[k],
                    )
                    for k in "DCLRE"
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        with st.expander("Linguistic feature details"):
            st.json(r.features)
    with tabs[2]:
        st.caption(
            "Hotel-corpus similarity is meaningful primarily for hotel-domain text. Similarity is not evidence of shared authorship."
        )
        if not r.neighbors:
            st.info("No comparison pool supplied.")
        for n in r.neighbors:
            with st.expander(
                f'Review {n["index"]+1} · semantic {n["similarity"]:.3f} · lexical {n.get("lexical_similarity",0):.3f}'
            ):
                st.write(n["text"])
                st.code(
                    word_diff(r.cleaned_text, n["text"]), language=None, wrap_lines=True
                )
    with tabs[3]:
        st.caption(
            "Signed TF-IDF × coefficient contributions from one calibrated linear estimator. Positive pushes toward deceptive; negative toward truthful. Not a causal explanation."
        )
        if not r.influential_terms:
            st.info("Text coefficients unavailable for this model.")
        for t in r.influential_terms:
            st.markdown(
                '<span class="term">'
                + html.escape(t["term"])
                + f' {t["contribution"]:+.3f}</span>',
                unsafe_allow_html=True,
            )
    with tabs[4]:
        st.text(r.text)
        st.divider()
        st.text(r.cleaned_text)
    a, b = st.columns(2)
    a.download_button(
        "↓ Download analysis JSON",
        json.dumps(r.to_dict(), indent=2),
        file_name="reviewguard-analysis.json",
        mime="application/json",
        key=key + "_json",
    )
    b.download_button(
        "↓ Download review PDF",
        pdf_bytes(r),
        file_name="reviewguard-review.pdf",
        mime="application/pdf",
        key=key + "_pdf",
    )
