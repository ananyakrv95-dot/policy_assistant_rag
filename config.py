from pathlib import Path


INDEX_FOLDER = Path("faiss_index")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DATA_FOLDER = Path("data")
LLM_MODEL = "llama-3.3-70b-versatile"

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
