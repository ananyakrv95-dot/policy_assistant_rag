from langchain_huggingface import HuggingFaceEmbedding
from langchain_community.vectorstores import FAISS
from config import (
    INDEX_FOLDER,
    EMBEDDING_MODEL
)

def create_embed_model():
    model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL
    )

    return model
    

def create_index(chunks):
    embed_model = create_embed_model()

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embed_model
    )
    vector_store.save_local(INDEX_FOLDER)
    
    return vector_store

