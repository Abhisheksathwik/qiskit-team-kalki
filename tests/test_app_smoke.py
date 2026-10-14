from streamlit.testing.v1 import AppTest


def test_app_initial_page_renders_without_exceptions():
    app = AppTest.from_file("app.py", default_timeout=20).run()
    assert not app.exception
    assert any("Run experiment" in button.label for button in app.button)
