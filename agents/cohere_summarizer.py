import os, sys
import time
import ray
import logging
import cohere

from typing import List
from textwrap import wrap

# Setup path and import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

# --- Logging setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "summarizer_cohere.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# --- Ray Init ---
if not ray.is_initialized():
    ray.init(num_cpus=4, ignore_reinit_error=True)

# --- Parameters ---
MODEL_NAME = "command-light"
MAX_TOKENS = 4096
SAFE_CHUNK_LENGTH = 1500  # characters, approx ~600–800 tokens
NUM_WORKERS = 4

logger.info(f"[INFO] Starting new test with model: {MODEL_NAME}")

# --- Ray actor for summarization ---
@ray.remote
class CohereSummarizer:
    def __init__(self, api_key):
        self.client = cohere.Client(api_key)

    def summarize(self, chunk: str) -> str:
        prompt = (
            "Summarize the following podcast transcript chunk. Focus on key points:\n\n"
            f"{chunk}\n\nSummary:"
        )
        response = self.client.chat(
            message=prompt,
            model=MODEL_NAME,
            temperature=0.3,
            max_tokens=300,
        )
        return response.text.strip()

# --- Chunking strategy (safe and simple) ---
def split_text(text: str, chunk_size: int) -> List[str]:
    return wrap(text, width=chunk_size, break_long_words=False, replace_whitespace=False)

# --- Main summarization pipeline ---
def summarize_text(text: str, chunk_size: int = SAFE_CHUNK_LENGTH, num_workers: int = NUM_WORKERS) -> str:
    start = time.time()

    chunks = split_text(text, chunk_size)
    logger.info(f"[START] Summarizing {len(chunks)} chunks with model `{MODEL_NAME}` using {num_workers} workers.")

    actors = [CohereSummarizer.remote(config.COHERE_API_KEY) for _ in range(min(num_workers, len(chunks)))]
    futures = [actors[i % len(actors)].summarize.remote(chunk) for i, chunk in enumerate(chunks)]
    summaries = ray.get(futures)

    final_summary = "\n\n".join(summaries)
    logger.info(f"[DONE] Cohere summarization completed in {time.time() - start:.2f} seconds.")
    logger.info(f"[INFO] Final output: {len(final_summary)} characters across {len(chunks)} chunks.")

    ray.shutdown()
    return final_summary

# --- Test entrypoint ---
if __name__ == "__main__":
    logger.info("[TEST] Running summarizer using Cohere API...")

    sample_text = (
        "SpaceX is preparing for a new Starship test flight. "
        "The rocket aims to reach orbital velocity and return safely to Earth. "
        "NASA plans to use the system for lunar missions. "
        "Elon Musk has said that interplanetary travel is the ultimate goal. "
    ) * 100  # simulate long transcript

    t1 = time.time()
    summary = summarize_text(sample_text)
    t2 = time.time()

    print(f"\n⏱️ Elapsed: {t2 - t1:.2f} seconds")
    print(f"\n📄 Final Summary Preview:\n{summary[:500]}...\n")
