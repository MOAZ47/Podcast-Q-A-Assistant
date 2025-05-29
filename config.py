import os

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "qsgmkeaskuouflhwsx8ra.c0.asia-southeast1.gcp.weaviate.cloud")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "td5y9thNzdjURYoUW6fCwlXIYxgUqOg4ddwm")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "F7aq79uHk5nq5vStWOxGDE6nGB5h545DIua5gJrV")
AUDIO_FILE_PATH = "elon_musk_starship_rocket.mp3"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
LLM_MODEL = "command"