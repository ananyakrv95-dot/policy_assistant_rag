from loader.langchain_documents_loader import load_documents
from config import DATA_FOLDER
import random
import hashlib
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from config import EMBEDDING_MODEL, LLM_MODEL
from ragas.testset import TestsetGenerator
import pandas as pd

load_dotenv()

documents = load_documents(DATA_FOLDER)

print("Total documents:", len(documents))
print(documents[0])

pdf_documents = [document for document in documents if str(document.metadata.get("source", "")).endswith(".pdf")]
csv_documents = [document for document in documents if "record_id" in document.metadata]

print("PDF pages:", len(pdf_documents))
print("CSV rows:", len(csv_documents))

random.seed(42)

csv_sample = random.sample(
    csv_documents,
    min(100, len(csv_documents)),
)

def get_file_hash(file_path: Path) -> str:
    """Compute the SHA256 hash of a file."""
    hasher = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_unique_pdf_paths(folder_path: Path) -> list[Path]:
    """Get unique PDF file paths based on their content hash."""
    unique_pdf_paths = []
    seen_hashes = set()

    for pdf_path in sorted(folder_path.rglob("*.pdf")):
        file_hash = get_file_hash(pdf_path)
        if file_hash in seen_hashes:
            print(f"Skipping duplicate: {pdf_path.name}")
            continue

        seen_hashes.add(file_hash)
        unique_pdf_paths.append(pdf_path)

    return unique_pdf_paths


def load_unique_pdf_documents(folder_path: Path):
    """Load unique PDF documents from a folder."""
    unique_pdf_documents = []
    unique_pdf_paths = get_unique_pdf_paths(folder_path)

    for pdf_path in unique_pdf_paths:
        pages = PyPDFLoader(str(pdf_path)).load_and_split()

        for page in pages:
            page.metadata["source_type"] = "pdf"
            page.metadata["source_file"] = pdf_path.name
            page.metadata["page_number"] = (
                page.metadata.get("page", 0) + 1
            )

        unique_pdf_documents.extend(pages)

    return unique_pdf_documents

pdf_documents = load_unique_pdf_documents(DATA_FOLDER)

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing.")

llm = ChatGroq(
    model=LLM_MODEL,
    temperature=0,
)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
)

generator = TestsetGenerator.from_langchain(
    llm=llm,
    embedding_model=embeddings,

)

pdf_testset = generator.generate_with_langchain_docs(
    documents=pdf_documents,
    testset_size=15,
)

csv_testset = generator.generate_with_langchain_docs(
    documents=csv_sample,
    testset_size=15,
)


pdf_dataframe = pdf_testset.to_pandas()
pdf_dataframe["source_group"] = "pdf"

csv_dataframe = csv_testset.to_pandas()
csv_dataframe["source_group"] = "csv"

candidates = pd.concat(
    [pdf_dataframe, csv_dataframe],
    ignore_index=True,
)

candidates.to_csv(
    "evaluation/datasets/candidates.csv",
    index=False,
)

