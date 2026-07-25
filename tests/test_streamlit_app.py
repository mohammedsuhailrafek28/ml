from streamlit.testing.v1 import AppTest
def test_app_loads_without_exception():
    app=AppTest.from_file('app.py').run(); assert not app.exception; assert len(app.sidebar.radio)==1
