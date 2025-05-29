import os

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
AUDIO_FILE_PATH = "elon_musk_starship_rocket.mp3"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
LLM_MODEL = "command"
