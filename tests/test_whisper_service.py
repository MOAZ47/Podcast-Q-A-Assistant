 
from services.whisper_service import transcribe_audio
import os

def test_transcribe_audio():
    test_audio_path = "example_audio/test_audio.mp3"
    assert os.path.exists(test_audio_path), "Test audio file missing!"
    text = transcribe_audio(test_audio_path)
    assert isinstance(text, str)
    assert len(text) > 10  # Ensure transcription is not empty
