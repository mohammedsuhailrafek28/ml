import pytest
from pathlib import Path

pytest.importorskip("streamlit", reason="legacy Streamlit UI is an optional dev dependency")

from streamlit.testing.v1 import AppTest  # noqa: E402


def test_app_loads_without_exception():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    assert not app.exception
    assert len(app.sidebar.radio) == 1
