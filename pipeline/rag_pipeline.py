import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from retriever.hybrid_retriever import (
    create_bm25_retriever,
    hybrid_retrieve,
    load_chunks,
)
from retriever.reranker import (
    create_cohere_reranker,
    rerank_documents,
)
from config import DATA_FOLDER, LLM_MODEL, PROMPT_TEMPLATE
from retriever.retriever import load_vector_store, retrieve_chunks

load_dotenv()

def format_context(documents: list[Document]) -> str:
    """
    Convert retrieved LangChain documents into text
    that can be passed to the LLM.
    """

    formatted_chunks = []

    for index,document in enumerate(documents, start=1):
        source = document.metadata.get(
            "source",
            document.metadata.get(
                "file_name", "Unknown source"
            ),
        )

        title = document.metadata.get(
            "article_title",
            document.metadata.get("file_name", "Untitled"),
        )

        formatted_chunk = (
            f"[Source {index}]\n"
            f"Title: {title}\n"
            f"Source: {source}\n"
            f"Content:\n{document.page_content}"
        )
            

        formatted_chunks.append(formatted_chunk)

    return "\n\n---\n\n".join(formatted_chunks)


def create_llm() -> ChatGroq :
    """
    Create the Groq Llama model.
    """
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )
    
    return ChatGroq(
        model=LLM_MODEL,
        temperature=0,
    )

"""def create_rag_pipeline():
    vector_store = load_vector_store()

    prompt = ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE
    )

    llm = create_llm()

    chain = prompt | llm

    return vector_store, chain
    """

    def create_rag_pipeline():
    vector_store = load_vector_store()

    chunks = load_chunks()
    bm25_retriever = create_bm25_retriever(
        chunks=chunks,
        k=10,
    )

    reranker = create_cohere_reranker()
    llm = create_llm()

    prompt = ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE
    )

    chain = prompt | llm

    return (
        vector_store,
        bm25_retriever,
        reranker,
        llm,
        chain,
    )

def build_sources(documents: list[Document]) -> list[dict]:
    sources = []
    for number, document in enumerate(documents, start=1):
        metadata = document.metadata
        sources.append(
            {
                "number": number,
                "title": metadata.get(
                    "article_title",
                    metadata.get("file_name", "Untitled"),
                ),
                "source": metadata.get(
                    "source",
                    metadata.get(
                        "file_name",
                        "Unknown source",
                    ),
                ),
                "content": document.page_content,
            }
        )
    return sources

"""def answer_question(
    vector_store,
    chain,
    question: str,
    k: int = 5,
):
    documents = retrieve_chunks(
        vector_store=vector_store,
        query=question,
        k=k,
    )

    context = format_context(documents)

    response = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return {
        "answer": response.content,
        "sources": build_sources(documents),
    }

"""

def answer_question(
    vector_store,
    bm25_retriever,
    reranker,
    llm,
    chain,
    question: str,
    mode: str = "hybrid_rerank",
    use_hyde: bool = False,
):
    dense_query = (
        create_hyde_query(llm, question)
        if use_hyde
        else question
    )

    if mode == "dense":
        documents = vector_store.similarity_search(
            dense_query,
            k=3,
        )

    elif mode == "hybrid":
        documents = hybrid_retrieve(
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
            query=question,
            dense_query=dense_query,
            k=10,
        )[:3]

    elif mode == "hybrid_rerank":
        candidates = hybrid_retrieve(
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
            query=question,
            dense_query=dense_query,
            k=10,
        )

        documents = rerank_documents(
            reranker=reranker,
            query=question,
            documents=candidates,
        )

    else:
        raise ValueError(f"Unknown retrieval mode: {mode}")

    context = format_context(documents)

    response = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return {
        "answer": response.content,
        "sources": build_sources(documents),
        "hyde_query": dense_query if use_hyde else None,
    }
