 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.init import Auth

from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME, WEAVIATE_API_KEY, WEAVIATE_URL, COHERE_API_KEY

def split_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "?", "!"]
    )
    return splitter.split_text(text)

def setup_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        headers={"X-Cohere-Api-Key": COHERE_API_KEY}
    )
    return WeaviateVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings,
        client=client,
        index_name="Podcast",
        by_text=False
    )