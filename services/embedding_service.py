 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
import weaviate
from weaviate.classes.init import Auth
import config

# You can extract keys and URL from environment or pass them as args
def create_vectorstore(text_chunks: list) -> weaviate:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=config.WEAVIATE_URL,
        auth_credentials=Auth.api_key(config.WEAVIATE_API_KEY),
        headers={"X-Cohere-Api-Key": config.COHERE_API_KEY}
    )

    vectorstore = WeaviateVectorStore.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        client=client,
        index_name="Podcast",
        by_text=False
    )

    return vectorstore
