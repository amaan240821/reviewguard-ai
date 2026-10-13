import sys, logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import streamlit as st

SHIELD = '<svg viewBox="0 0 40 44" fill="none"><path d="M20 2 36 8v12c0 10-8 17-16 21C12 37 4 30 4 20V8L20 2Z" stroke="#52e0db" stroke-width="2"/><circle cx="19" cy="20" r="7" stroke="#a78bfa" stroke-width="2"/><path d="m24 25 6 6" stroke="#a78bfa" stroke-width="2"/></svg>'


def setup():
    st.set_page_config(
        page_title="ReviewGuard AI · Review intelligence", page_icon="🛡️", layout="wide"
    )
    st.markdown(
        "<style>" + (ROOT / "app/ui/styles.css").read_text() + "</style>",
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.markdown(
            '<div class="brand">'
            + SHIELD
            + 'ReviewGuard <small>AI</small></div><p class="muted" style="font-size:.72rem">REVIEW INTELLIGENCE WORKSPACE</p>',
            unsafe_allow_html=True,
        )
        if not (ROOT / "data/raw/deceptive-opinion.csv").exists():
            st.warning("Dataset missing · run make data")
        if not (ROOT / "models/deception_clf.joblib").exists():
            st.warning("Classifier missing · run make train")


def header(kicker, title, subtitle):
    st.markdown(f'<div class="eyebrow">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="subhead">{subtitle}</div>', unsafe_allow_html=True)


def disclaimer():
    st.markdown(
        '<div class="disclaimer">SCREENING, NOT CERTAINTY · ReviewGuard identifies suspicious patterns. A score cannot establish whether a review is truthful or whether authors coordinated. Always review the evidence in context.</div>',
        unsafe_allow_html=True,
    )


def safe_error(exc):
    logging.getLogger(__name__).exception("User operation failed")
    st.error(
        str(exc)
        if isinstance(exc, (ValueError, RuntimeError, FileNotFoundError))
        else "This operation could not finish. Check the input and local model files, then try again."
    )
