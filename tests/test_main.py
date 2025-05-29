# test_main.py
import pytest
from main import transcribe_audio

def test_transcribe_audio_with_sample_file():
    result = transcribe_audio(r"C:\Users\moazh\AppData\Local\Temp\tmpxpgbxuro.mp3")
    assert isinstance(result, str)
    assert "expected text snippet" in result  # Replace with actual snippet
