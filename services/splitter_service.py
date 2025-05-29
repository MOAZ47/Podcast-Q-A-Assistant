 
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

def split_text(transcription: str, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "?", "!"]
    )
    return text_splitter.split_text(transcription)
