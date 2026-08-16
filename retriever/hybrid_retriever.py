import hashlib
import pickle

from config import (
    CHUNKS_FILE,
    HYBRID_BM25_WEIGHT,
    HYBRID_FAISS_WEIGHT,
)

def document_id(document: Document) -> str:
    source = str(document.metadata.get("source", ""))
    page = str(document.metadata.get("page", ""))
    content = document.page_content

    value = f"{source}|{page}|{content}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_chunks() -> list[Document]:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            "Saved chunks were not found. Run: python ingest.py"
        )

    with CHUNKS_FILE.open("rb") as file:
        return pickle.load(file)

def create_bm25_retriever(
    chunks: list[Document],
    k: int = 10,
) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k
    return retriever


def rank_scores(documents: list[Document]) -> dict[str, float]:
    """
    Convert rank position into a 0–1 score.

    First result receives 1.0; later results receive
    progressively smaller values.
    """
    total = len(documents)

    if total == 0:
        return {}

    if total == 1:
        return {document_id(documents[0]): 1.0}

    return {
        document_id(document): 1 - (index / (total - 1))
        for index, document in enumerate(documents)
    }


def hybrid_retrieve(
    vector_store,
    bm25_retriever: BM25Retriever,
    query: str,
    dense_query: str | None = None,
    k: int = 10,
    bm25_weight: float = HYBRID_BM25_WEIGHT,
    faiss_weight: float = HYBRID_FAISS_WEIGHT,
) -> list[Document]:
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    # Original query preserves exact keywords for BM25.
    bm25_documents = bm25_retriever.invoke(query)[:k]

    # HyDE text can optionally improve semantic FAISS retrieval.
    faiss_documents = vector_store.similarity_search(
        dense_query or query,
        k=k,
    )

    bm25_scores = rank_scores(bm25_documents)
    faiss_scores = rank_scores(faiss_documents)

    documents_by_id = {
        document_id(document): document
        for document in bm25_documents + faiss_documents
    }

    combined = []

    for doc_id, document in documents_by_id.items():
        bm25_score = bm25_scores.get(doc_id, 0.0)
        faiss_score = faiss_scores.get(doc_id, 0.0)

        hybrid_score = (
            bm25_weight * bm25_score
            + faiss_weight * faiss_score
        )

        document.metadata["bm25_score"] = bm25_score
        document.metadata["faiss_score"] = faiss_score
        document.metadata["hybrid_score"] = hybrid_score

        combined.append(document)

    combined.sort(
        key=lambda document: document.metadata["hybrid_score"],
        reverse=True,
    )

    return combined[:k]
