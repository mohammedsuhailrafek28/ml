import pytest

pytest.importorskip("streamlit", reason="legacy Streamlit UI is an optional dev dependency")

from streamlit.testing.v1 import AppTest  # noqa: E402


def test_app_loads_without_exception():
    app = AppTest.from_file("app.py").run()
    assert not app.exception
    assert len(app.sidebar.radio) == 1
