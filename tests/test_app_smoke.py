from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_app_initial_page_renders_without_exceptions():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=20).run()
    assert not app.exception
    assert any("Run experiment" in button.label for button in app.button)



def test_classical_only_experiment_renders_results_page():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=20).run()
    for checkbox in app.checkbox:
        if checkbox.label in ("Hybrid VQC", "Quantum Kernel + SVM"):
            checkbox.set_value(False)
    app.run()
    run_button = next(button for button in app.button if "Run experiment" in button.label)
    run_button.click()
    app.run()
    assert not app.exception
    assert any("Experiment scoreboard" in heading.value for heading in app.subheader)
    assert any("Classical RBF SVM" in str(frame.value) for frame in app.dataframe)
