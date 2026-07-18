from pathlib import Path
from langchain_hugginface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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