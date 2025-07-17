import os
import time
import ray
import logging
from transformers import pipeline

# --- Logging Setup ---
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_file_dir, ".."))
LOG_DIR = os.path.join(project_root, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, "summarizer_events.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Model Config ---
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

logger.info(f"[INFO] Starting new test with {MODEL_NAME}")

# --- Ray Init ---
start_time = time.time()
if not ray.is_initialized():
    ray_init_start = time.time()
    ray.init(num_cpus=4, ignore_reinit_error=True)
    ray_init_end = time.time()
    logger.info(f"[INFO] Ray initialized in {ray_init_end - ray_init_start:.2f} seconds.")



# --- Ray Actor for Summarization ---
@ray.remote
class SummarizerActor:
    def __init__(self):
        model_load_start = time.time()
        self.summarizer = pipeline("summarization", model=MODEL_NAME)
        model_load_end = time.time()
        logger.info(f"[INFO] Model loaded inside actor in {model_load_end - model_load_start:.2f} seconds.")

    def summarize(self, text):
        return self.summarizer(text)[0]["summary_text"]

# --- Main summarization function ---
def summarize_text(text: str, chunk_size: int = 1000, num_workers: int = 4, model_name: str = MODEL_NAME) -> str:
    total_start = time.time()

    # Split text into chunks
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    logger.info(f"[START] Summarization with {len(chunks)} chunks using `{model_name}` and {num_workers} workers.")

    # Create Ray actors
    actors = [SummarizerActor.remote() for _ in range(min(num_workers, len(chunks)))]

    # Assign chunks to actors round-robin
    futures = [actors[i % len(actors)].summarize.remote(chunk) for i, chunk in enumerate(chunks)]

    # Collect summaries
    summaries = ray.get(futures)

    total_end = time.time()
    logger.info(f"[DONE] Summarization completed in {total_end - total_start:.2f} seconds.")
    logger.info(f"[INFO] Final summary length: {len(' '.join(summaries))} characters.")

    ray.shutdown()  # Clean shutdown

    return " ".join(summaries)

# --- Test block ---
if __name__ == "__main__":
    logger.info("[TEST] Running summarizer on simulated text...")

    # Fake long transcript
    sample_text = (
        "SpaceX is preparing for a new Starship test flight. "
        "The rocket aims to reach orbital velocity and return safely to Earth. "
        "NASA plans to use the system for lunar missions. "
        "Elon Musk has said that interplanetary travel is the ultimate goal. "
    ) * 100  # simulate long transcript

    t_start = time.time()
    result = summarize_text(sample_text)
    t_end = time.time()

    print(f"\n⏱️ Elapsed: {t_end - t_start:.2f} seconds")
    print(f"\n📄 Summary Preview:\n{result[:500]}...\n")
