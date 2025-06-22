import whisper


def transcribe_podcast(file_path: str) -> str:
    model = whisper.load_model("base")  # "small", "medium", or "large" for better accuracy
    result = model.transcribe(file_path)
    return result["text"]
