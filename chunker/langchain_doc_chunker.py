from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from config import (
    INDEX_FOLDER,
    EMBEDDING_MODEL
)


def get_document_chunks(documents, chunk_size=500, chunk_overlap=100):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return text_splitter.split_documents(documents)

def get_document_chunks_using_tokens(documents, chunk_size=500, chunk_overlap=100):
    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL
    )

    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return splitter.split_documents(documents)
