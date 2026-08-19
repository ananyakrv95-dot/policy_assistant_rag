from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config import (
    EMBEDDING_MODEL
)

def create_embed_model():
    model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL
    )

    return model