from pathlib import Path
from llama_index.core import SimpleDirectoryReader, Document


def load_documents(folder_path: Path) -> list[Document]:
    documents = SimpleDirectoryReader(
        input_dir=str(folder_path),
        required_exts=[".pdf",".txt",".csv"],
        recursive=True,
    )

    return documents.load_data()




