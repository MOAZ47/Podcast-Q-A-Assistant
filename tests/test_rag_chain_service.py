
from services.rag_chain_service import build_rag_chain
from services.embedding_service import create_vectorstore

def test_rag_chain_inference():
    chunks = ["Apple released a new iPhone.", "Google introduced Gemini model."]
    vectorstore = create_vectorstore(chunks)
    chain = build_rag_chain(vectorstore)
    question = "What did Apple do?"
    response = chain.invoke(question)
    assert isinstance(response, str)
    assert len(response) > 5
