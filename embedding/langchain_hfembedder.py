from langchain_huggingface import HuggingFaceEmbedding

def create_embed_model():
    model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    return model
    