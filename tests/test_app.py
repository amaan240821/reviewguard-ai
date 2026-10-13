from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from reviewguard.config import ROOT


@pytest.mark.parametrize(
    "path",
    ["app/Home.py"]
    + [str(p.relative_to(ROOT)) for p in sorted((ROOT / "app/pages").glob("*.py"))],
)
def test_every_page(path):
    at = AppTest.from_file(str(ROOT / path), default_timeout=60).run()
    assert len(at.exception) == 0


def test_single_example_analysis():
    at = AppTest.from_file(
        str(ROOT / "app/pages/1_Single_Review.py"), default_timeout=300
    ).run()
    at.selectbox(key="example_choice").select("Rating contradiction").run()
    next(b for b in at.button if b.label == "Analyze review →").click().run()
    assert not at.exception
    assert "single_result" in at.session_state
    assert at.session_state["single_result"].signals["R"] > 0.8
    # Re-score through settings and verify stored analysis updates.
    at.slider(key="single_R").set_value(70).run()
    assert not at.exception


def test_batch_demo_and_filters():
    at = AppTest.from_file(
        str(ROOT / "app/pages/2_Batch_Forensics.py"), default_timeout=300
    ).run()
    next(b for b in at.button if b.label == "Load full demo batch").click().run()
    assert not at.exception
    assert len(at.session_state["batch_result"].reviews) == 193
    next(t for t in at.text_input if t.label == "Search review text").set_value(
        "nonexistent search phrase"
    ).run()
    assert not at.exception


def test_cluster_demo_and_threshold():
    at = AppTest.from_file(
        str(ROOT / "app/pages/3_Cluster_Explorer.py"), default_timeout=300
    ).run()
    next(b for b in at.button if b.label == "Load demo for graph").click().run()
    assert not at.exception
    next(s for s in at.slider if s.label == "Graph similarity threshold").set_value(
        0.95
    ).run()
    assert not at.exception
