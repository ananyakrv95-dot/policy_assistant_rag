import csv
from pathlib import Path
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    CSVLoader,
    PyPDFLoader
)




def validate_csv(file_path: Path):
    
    with file_path.open(mode="r",encoding="utf-8-sig",newline="",) as file:
        reader = csv.DictReader(file)
        """
        Suppose the CSV looks like:
        record_id,retrieval_context,article_title
        1,This policy covers health insurance,Health Policy

        DictReader converts the row into:

        {
            "record_id": "1",
            "retrieval_context": "This policy covers health insurance",
            "article_title": "Health Policy"
        }
        """


        #check whether required column exists

        # Convert field names to a set for fast lookup and deduplication, defaulting to an empty list if None
        available_columns = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - available_columns

        if missing_columns:
            raise ValueError(
                f"{file_path.name} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        for row_number,row in enumerate(reader, start=2):
            context = row["retrieval_context"].strip()

            if not context:
                raise ValueError(
                    f"Empty retrieval_context in "
                    f"{file_path.name}, row {row_number}"
                )
    print(f"CSV validated: {file_path.name}")


"""def create_csv_loader(file_path: Path):

    # Only this content_columns will be embedded
    file_path_str=str(file_path)
    return CSVLoader(
        file_path=file_path_str,
        encoding="utf-8-sig",
        content_columns=["retrieval_context"],
        metadata_columns=[
            "record_id",
            "dataset_name",
            "article_title",
            "paragraph_index",
        ],
        source_column="source_url",
    )"""
# Map file extensions to their specific LangChain loaders

"""extension_loader_mapping = {
    ".txt": lambda path: TextLoader(path, encoding="utf-8", autodetect_encoding=True,),
    ".pdf": lambda path: PyPDFLoader(path),
    ".csv": lambda path: CSVLoader(path)
}"""


"""A LangChain document looks roughly like:

Document(
    page_content="Actual searchable text",
    metadata={"source": "policy.pdf"}
)"""


def load_documents(folder_path: Path):
    """
    Validate every CSV before LangChain loads it
    """
    for csv_file in folder_path.rglob("*.csv"):
        validate_csv(csv_file)

    """
    Load all PDF files
    """
    pdf_loader = DirectoryLoader(
        path=str(folder_path),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )

    """
    Load all Text files
    """
    txt_loader = DirectoryLoader(
        path=str(folder_path),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={
            encoding="utf-8", 
            autodetect_encoding=True
        },
    )

    """
    Load all CSV files
    """
    csv_loader = DirectoryLoader(
        path=str(folder_path),
        glob="**/*.csv",
        loader_cls=CSVLoader,
        loader_kwargs={
            "encoding"="utf-8-sig",
            # Searchable text
            "content_columns": [
                "retrieval_context",
            ],

            # Supporting information
            "metadata_columns": [
                "record_id",
                "dataset_name",
                "article_title",
                "paragraph_index",
            ],

            # Stored as metadata["source"]
            "source_column": "source_url",
        },
    )

    pdf_doc = pdf_loader.load()
    txt_doc = txt_loader.load()
    csv_doc = csv_loader.load()

    # Combine all LangChain Document objects

    all_documents = (
        pdf_doc
        + txt_doc
        + csv_doc
    )

    print(f"PDF documents loaded: {len(pdf_doc)}")
    print(f"TXT documents loaded: {len(txt_doc)}")
    print(f"CSV documents loaded: {len(csv_doc)}")
    
    print(f"Total documents loaded: {len(all_documents)}")

    return all_documents