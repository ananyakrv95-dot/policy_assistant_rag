import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from config import DATA_FOLDER, LLM_MODEL, PROMPT_TEMPLATE
from retriever.retriever import load_vector_store, retrieve_chunks

load_dotenv()

def format_context(documents: list[Document]) -> str:
    """
    Convert retrieved LangChain documents into text
    that can be passed to the LLM.
    """

    formatted_chunks = []

    for index,document in enumerate(documents, start=1):
        source = documents.metadata.get(
            "source",
            document.metadata.get(
                "file_name", "Unknown source"
            ),
        )

        title = documents.metadata.get(
            "article_title",
            document.metadata.get("file_name", "Untitled"),
        )

        formatted_chunk = 
            f"[Source {index}]\n"
            f"Title: {title}\n"
            f"Source: {source}\n"
            f"Content:\n{document.page_content}"

        formatted_chunks.append(formatted_chunk)

    return "\n\n---\n\n".join(formatted_chunks)


def create_llm() -> ChatGroq :
    """
    Create the Groq Llama model.
    """
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )
    
    return ChatGroq(
        model=LLM_MODEL,
        temperature=0,
    )

def create_rag_pipeline():
    