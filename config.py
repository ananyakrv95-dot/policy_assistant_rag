from pathlib import Path


INDEX_FOLDER = "faiss_index"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DATA_FOLDER = "data"

REQUIRED_COLUMNS = {
    "retrieval_context",
    "record_id",
    "dataset_name",
    "article_title",
    "paragraph_index",
    "source_url",
}
