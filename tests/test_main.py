import pytest
import os
import logging
from unittest.mock import patch, MagicMock

from main import initialize_pipeline, LOG_PATH

# --- Fixtures ---
# Yeh fixture har test se pehle aur baad mein run hoga
# Logging ko temporary disable ya reset karne ke liye useful
@pytest.fixture(autouse=True)
def setup_teardown_logging():
    # Test se pehle: existing handlers ko hata do taaki test logs clean rahein
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:] # Copy existing handlers
    root_logger.handlers = [] # Clear them

    # main.py ke logger ko reset karein
    main_logger = logging.getLogger('main')
    main_logger.handlers = []
    main_logger.propagate = True # Ensure events are passed to root logger

    # Test run hone dein
    yield

    # Test ke baad: original handlers ko wapas add kar do
    root_logger.handlers = original_handlers
    # Clear main.log file after each test run to keep it clean (optional)
    if os.path.exists(LOG_PATH):
        try:
            os.remove(LOG_PATH)
        except OSError as e:
            print(f"Error removing log file {LOG_PATH}: {e}")

# Sample transcript
@pytest.fixture
def sample_transcript():
    return (
        "SpaceX Starship test launch happened. "
        "The booster did not relight and crashed. "
        "Starship made it to space but had a fuel leak and burned up. "
        "Previous launches exploded before reaching space. "
        "NASA wants Starship to land people on the moon by 2027. "
        "Elon Musk hopes Starship will carry people to Mars. "
        "Refueling in space is a challenge."
    )

# --- Tests ---

# Test 1: Basic pipeline flow with valid input
def test_initialize_pipeline_success(sample_transcript):
    # `patch` decorators ka use karke imported functions ko mock karenge
    # Isse hum un functions ke actual execution ko bypass kar sakte hain
    # aur unke return values ko control kar sakte hain.
    with patch('main.summarize_text') as mock_summarize, \
         patch('main.fact_check') as mock_fact_check, \
         patch('main.generate_final_report') as mock_generate_report:

        # Mock functions ke return values set karein
        mock_summarize.return_value = "This is a summarized text."
        mock_fact_check.return_value = {"fact1": True, "fact2": False}
        mock_generate_report.return_value = "This is the final generated report."

        # Pipeline ko run karein
        report = initialize_pipeline(sample_transcript)

        # Verify karein ki har function sahi arguments ke saath call hua
        mock_summarize.assert_called_once_with(sample_transcript)
        mock_fact_check.assert_called_once_with("This is a summarized text.")
        mock_generate_report.assert_called_once_with("This is a summarized text.", {"fact1": True, "fact2": False})

        # Verify karein ki function ka final output sahi hai
        assert "This is the final generated report." in report
        assert "Error" not in report # Ensure no error message


