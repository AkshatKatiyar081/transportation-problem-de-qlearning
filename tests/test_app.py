"""Smoke test of the dashboard: every example loads and every applicable method runs without errors."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.mark.parametrize("name", ["A5", "N5", "X1", "X2", "X3", "X4", "U1"])
def test_app_runs(name):
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.selectbox(key="source").set_value(name).run()
    assert not at.exception
    for m in ["modi", "milp", "naive", "hybrid"]:
        at.sidebar.radio(key="method").set_value(m).run()
        run_btn = [b for b in at.sidebar.button if "RUN" in b.label][0]
        if not run_btn.disabled:
            if m == "hybrid":
                at.sidebar.number_input[0].set_value(2000).run()
            run_btn = [b for b in at.sidebar.button if "RUN" in b.label][0]
            run_btn.click().run()
        assert not at.exception, [e.value for e in at.exception]


def test_custom_builder_loads():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.selectbox(key="source").set_value("__custom__").run()
    assert not at.exception
    run_btn = [b for b in at.sidebar.button if "RUN" in b.label][0]
    run_btn.click().run()
    assert not at.exception
