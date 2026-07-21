from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import (
    INDEX_FOLDER,
    EMBEDDING_MODEL
)


def load_vector_store():
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    return FAISS.load_local(
        folder_path=str(INDEX_FOLDER),
        embeddings=embedding,
    )

def retrieve_chunks(vector_store, query:str, k:int=5):

    """
    Return the top-k chunks related to the query.
    """
    if not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )
    
    documents = vector_store.similarity_search(
        query=query,
        k=k
    )

    return documents