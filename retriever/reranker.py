import os

from langchain_cohere import CohereRerank
from langchain_core.documents import Document

from config import COHERE_RERANK_MODEL, RERANK_TOP_N


def create_cohere_reranker() -> CohereRerank:
    if not os.getenv("COHERE_API_KEY"):
        raise ValueError(
            "COHERE_API_KEY is missing."
        )

    return CohereRerank(
        model=COHERE_RERANK_MODEL,
        top_n=RERANK_TOP_N,
    )


def rerank_documents(
    reranker: CohereRerank,
    query: str,
    documents: list[Document],
) -> list[Document]:
    return reranker.compress_documents(
        documents=documents,
        query=query,
    )