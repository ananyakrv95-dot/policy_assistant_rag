from pathlib import Path



INDEX_FOLDER = Path("faiss_index")
CHUNKS_FILE = INDEX_FOLDER / "chunks.pkl"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DATA_FOLDER = Path("data")
LLM_MODEL = "llama-3.3-70b-versatile"
HYBRID_BM25_WEIGHT = 0.4
HYBRID_FAISS_WEIGHT = 0.6

HYBRID_CANDIDATES = 10
RERANK_TOP_N = 3
COHERE_RERANK_MODEL = "rerank-v3.5"

REQUIRED_COLUMNS = {
    "retrieval_context",
    "record_id",
    "dataset_name",
    "article_title",
    "paragraph_index",
    "source_url",
}

PROMPT_TEMPLATE = """
You are conversation agent providing information.
Answer the question using only the context provided below.

Rules:
- Do not use outside knowledge.
- If the answer is unavailable in the context, say:
  "I could not find this information in the provided documents."
- Cite supporting chunks as [Source 1], [Source 2], etc.
- Give a clear and concise answer.

Context:
{context}

Question:
{question}

Answer:

"""
