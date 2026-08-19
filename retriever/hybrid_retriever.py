import hashlib
import pickle

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from config import (
    CHUNKS_FILE,
    HYBRID_BM25_WEIGHT,
    HYBRID_FAISS_WEIGHT,
)

def document_id(document: Document) -> str:
    """Create a stable ID for matching the same document."""
    source = str(document.metadata.get("source", ""))
    page = str(document.metadata.get("page", ""))
    content = document.page_content

    value = f"{source}|{page}|{content}"
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def load_chunks() -> list[Document]:
    """Load document chunks from a pickle file."""
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_FILE}. Run python ingest.py to create it.")
    
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
        return chunks
    

def create_bm25_retriever(
    chunks: list[Document],
    k: int = 10
) -> BM25Retriever:
    """Create a BM25 retriever from document chunks."""
    retriever = BM25Retriever.from_documents(
        chunks
    )
    retriever.k = k
    return retriever

    

"""def rank_scores(documents: list[Document]) -> dict[str, float]:
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
"""

def weighted_rrf(
    ranked_documents: list[tuple[list[Document], float]],
    rrf_k: int = 60, # List of (documents, weight) tuplesn
) -> list[Document]:
     """
    Combine multiple ranked result lists using weighted RRF.

    RRF score:
        weight / (rrf_k + rank)

    Rank starts from 1.
    """
    documents_by_id = {}
    fused_scores = {}

    for documents, weight in ranked_documents:
        for rank, document in enumerate(documents, start=1):
            doc_id = document_id(document)
            documents_by_id[doc_id] = document
            fused_scores[doc_id] = (
                fused_scores.get(doc_id, 0.0)
                + weight / (rrf_k + rank)
            )

    fused_documents = []

    for doc_id, document in documents_by_id.items():
        # Copy instead of mutating the original document.
        fused_document = Document(
            page_content=document.page_content,
            metadata={
                **document.metadata,
                "rrf_score": fused_scores[doc_id],
            },
        )

        fused_documents.append(fused_document)

    return sorted(
        fused_documents,
        key=lambda document: document.metadata["rrf_score"],
        reverse=True,
    )
        

def hybrid_retrieve(
    vector_store,
    bm25_retriever: BM25Retriever,
    query: str,
    dense_query: str | None = None,
    k: int = 10,
    bm25_weight: float = HYBRID_BM25_WEIGHT,
    faiss_weight: float = HYBRID_FAISS_WEIGHT,
) -> list[Document]:
    """Retrieve documents using a hybrid approach of BM25 and FAISS."""
    if not query.strip():
        raise ValueError("Query must not be empty.")

    if bm25_weight <= 0: 
        raise ValueError("At least one of bm25_weight must be greater than 0.")

    if faiss_weight <= 0:
        raise ValueError("At least one of faiss_weight must be greater than 0.")

    bm25_documents = bm25_retriever.invoke(query)[:k]
    faiss_documents = vector_store.similarity_search(
        dense_query or query,
        k=k
    )
    
    fused_docs = weighted_rrf(
        ranked_documents=[
            (bm25_documents, bm25_weight),
            (faiss_documents, faiss_weight),
        ],
        rrf_k=rrf_k,
    )

    return fused_docs[:k]
    
