from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_app_initial_page_renders_without_exceptions():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=20).run()
    assert not app.exception
    assert any("Run experiment" in button.label for button in app.button)
