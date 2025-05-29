 
from services.splitter_service import split_text

def test_split_text():
    long_text = "This is a test sentence. " * 50
    chunks = split_text(long_text, chunk_size=100, chunk_overlap=10)
    assert isinstance(chunks, list)
    assert len(chunks) > 1
    assert all(isinstance(chunk, str) for chunk in chunks)
