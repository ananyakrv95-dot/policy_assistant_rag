from config import DATA_FOLDER
from loader.langchain_documents_loader import load_documents
from chunker.langchain_doc_chunker import get_document_chunks_using_tokens
from embedding.embedandindex import create_index

def main():
    documents = load_documents(DATA_FOLDER)

    chunks = get_document_chunks_using_tokens(
        documents,
        chunk_size=500,
        chunk_overlap=100,
    )

    create_index(chunks)
    print("FAISS index created successfully.")

if __name__ == "__main__":
    main()

    

