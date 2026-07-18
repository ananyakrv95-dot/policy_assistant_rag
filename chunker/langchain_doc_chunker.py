from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_document_chunks(documents, chunk_size=500, chunk_overlap=100):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return text_splitter.split_documents(documents)