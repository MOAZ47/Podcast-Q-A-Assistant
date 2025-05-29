from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere import ChatCohere
import config

def build_rag_chain(vectorstore, query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """
    You are an AI assistant analyzing podcast conversations. Answer questions based on:
    Podcast Context: {sources}

    Question: {question}

    Provide:
    1. Direct answer (if found in context)
    2. Key points discussed (if relevant)
    3. Speaker quotes (if available)
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatCohere(cohere_api_key=config.COHERE_API_KEY, model="command")

    rag_chain = (
        {
            "sources": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(query)
