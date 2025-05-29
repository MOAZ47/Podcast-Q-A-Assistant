 
from services.whisper_service import transcribe_audio
from services.splitter_service import split_text
from services.embedding_service import create_vectorstore
from services.rag_chain_service import build_rag_chain

def initialize_pipeline(transcription: str, query: str):
    # 1. Transcribe
    #transcription = transcribe_audio(audio_path)

    # 2. Split into chunks
    chunks = split_text(transcription)

    # 3. Create vectorstore
    vectorstore = create_vectorstore(chunks)

    # 4. Build RAG chain
    #rag_chain = build_rag_chain(vectorstore)

    #return rag_chain, transcription
    return build_rag_chain(vectorstore, query)