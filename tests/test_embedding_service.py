 
from services.embedding_service import create_vectorstore

def test_create_vectorstore():
    sample_chunks = ["This is the first chunk.", "Another piece of text."]
    vectorstore = create_vectorstore(sample_chunks)
    assert vectorstore is not None
    retriever = vectorstore.as_retriever()
    results = retriever.get_relevant_documents("first")
    assert isinstance(results, list)
