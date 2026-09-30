"""Smoke test of the dashboard: every example loads and every applicable method runs without errors."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.mark.parametrize("name", ["A5", "S1", "X1", "X2", "X3", "X4", "U1"])
def test_app_runs(name):
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.selectbox(key="source").set_value(name).run()
    assert not at.exception
    for m in ["modi", "naive", "de", "hybrid"]:
        at.sidebar.radio(key="method").set_value(m).run()
        run_btn = [b for b in at.sidebar.button if "RUN" in b.label][0]
        if not run_btn.disabled:
            if m in ("de", "hybrid"):
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


def test_comparison_tab():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.sidebar.selectbox(key="source").set_value("S2").run()
    [w for w in at.number_input if (w.key or "").startswith("cmp_budget")][0].set_value(1000).run()
    at.number_input(key="cmp_n").set_value(2).run()
    at.button(key="cmp_go").click().run()
    assert not at.exception, [e.value for e in at.exception]
    log = list(at.session_state.runlog.values())[0]
    assert len(log) == 6          # 3 DE methods x 2 seeds


def test_pareto_and_risk():
    for name, btn in (("X3", "par_go"), ("X4", "risk_go")):
        at = AppTest.from_file(APP, default_timeout=300)
        at.run()
        at.sidebar.selectbox(key="source").set_value(name).run()
        key = "par_budget" if name == "X3" else "risk_budget"
        at.number_input(key=key).set_value(1000).run()
        if name == "X3":
            at.slider(key="par_n").set_value(3).run()
        at.button(key=btn).click().run()
        assert not at.exception, [e.value for e in at.exception]
