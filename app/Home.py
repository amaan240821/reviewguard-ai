import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
from app.ui.theme import setup
from app.ui.views import home

setup()
pages = [
    st.Page(home, title="Overview", icon=":material/dashboard:", default=True),
    st.Page(
        "pages/1_Single_Review.py", title="Single Review", icon=":material/search:"
    ),
    st.Page(
        "pages/2_Batch_Forensics.py",
        title="Batch Forensics",
        icon=":material/table_chart:",
    ),
    st.Page(
        "pages/3_Cluster_Explorer.py", title="Cluster Explorer", icon=":material/hub:"
    ),
    st.Page(
        "pages/4_Model_Evaluation.py",
        title="Model & Evaluation",
        icon=":material/monitoring:",
    ),
    st.Page(
        "pages/5_Dataset_Explorer.py",
        title="Dataset Explorer",
        icon=":material/database:",
    ),
    st.Page(
        "pages/6_About_Limitations.py",
        title="About & Limitations",
        icon=":material/info:",
    ),
]
nav = st.navigation(pages, position="hidden")
with st.sidebar:
    for page in pages:
        st.page_link(page)
    st.markdown(
        '<div class="side-foot">v1.0.0 · CPU inference<br>Dataset: '
        + (
            "ready"
            if (
                Path(__file__).resolve().parents[1] / "data/raw/deceptive-opinion.csv"
            ).exists()
            else "missing"
        )
        + "<br>Default: MiniLM + DistilBERT<br>Fallback: LSA + VADER<br>Local analysis · no review API</div>",
        unsafe_allow_html=True,
    )
nav.run()
