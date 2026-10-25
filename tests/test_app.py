"""Run the Streamlit app headless: it must render without network access or errors."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_app_renders_backtest_and_about():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Delhi-NCR PM2.5" in t.value for t in at.title)
    # Backtest tab: switching test period and station re-renders cleanly.
    at.radio(key="bt_fold").set_value("main").run()
    at.selectbox(key="bt_station").set_value(at.selectbox(key="bt_station").options[3]).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.dataframe) >= 1
