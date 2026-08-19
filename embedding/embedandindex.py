import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import (
    INDEX_FOLDER,
    EMBEDDING_MODEL,
    CHUNKS_FILE,
)

def create_embed_model():
    model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return model
    

"""def create_index(chunks):
    embed_model = create_embed_model()

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embed_model
    )
    vector_store.save_local(INDEX_FOLDER)
    
    return vector_store
    """

    def create_index(chunks):
        INDEX_FOLDER.mkdir(parents=True, exist_ok=True)
        embed_model = create_embed_model()

        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embed_model,
        )

        vector_store.save_local(str(INDEX_FOLDER))

        with CHUNKS_FILE.open("wb") as f:
            pickle.dump(chunks, f)

        return vector_store

