import os
import time
import logging
from faster_whisper import WhisperModel

# --- Logging ---
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "transcription.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Model Config ---
MODEL_SIZE = "base"  # small | medium | large-v2 | tiny
USE_GPU = False  # Set True for GPU if available
COMPUTE_TYPE = "int8" if not USE_GPU else "float16"

# --- Load model ---
def load_model(model_size = MODEL_SIZE):
    start_model = time.time()
    model = WhisperModel(MODEL_SIZE, device="cuda" if USE_GPU else "cpu", compute_type=COMPUTE_TYPE)
    logger.info(f"[INFO] Loaded Whisper model '{MODEL_SIZE}' on {'GPU' if USE_GPU else 'CPU'} in {time.time() - start_model:.2f}s")
    return model

# --- Transcription Function ---
def transcribe_audio(file_path: str, model_size = MODEL_SIZE) -> str:
    model = load_model(model_size)
    
    if not os.path.exists(file_path):
        logger.error(f"[ERROR] File not found: {file_path}")
        return ""

    logger.info(f"[START] Transcribing file: {file_path}")
    start = time.time()

    segments, info = model.transcribe(file_path, beam_size=5, language="en")
    transcript = " ".join([seg.text.strip() for seg in segments])

    end = time.time()
    logger.info(f"[DONE] Transcription completed in {end - start:.2f}s")
    logger.info(f"[INFO] Duration: {info.duration:.2f}s | Language: {info.language} | Segments: {len(transcript.split('.'))}")

    return transcript

# --- Test Entry Point ---
if __name__ == "__main__":
    SAMPLE_FILE = "elon_musk_starship_rocket.mp3"  # replace with your test file
    output = transcribe_audio(SAMPLE_FILE)
    print("\n🔍 Transcript Preview:\n", output[:1000], "...\n")
