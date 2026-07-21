import streamlit as st

from pipeline.rag_pipeline import (
    answer_question,
    create_rag_pipeline,
)

st.set_page_config(
    page_title="Chat Assistant",
    page_icon="📘",
    layout="centered",
)

st.title("Chat Assistant")
st.caption(
    "Ask questions based on the indexed NeGD PDFs and CSV data."
)


@st.cache_resource
def initialize_pipeline():
    return create_rag_pipeline()


try:
    vector_store, chain = initialize_pipeline()
except Exception as error:
    st.error(f"Could not initialize the RAG pipeline: {error}")
    st.stop()

# Configurable k
k = st.sidebar.slider(
    label="Number of retrieved chunks",
    min_value=1,
    max_value=10,
    value=5,
    step=1
)

st.sidebar.caption(
    "A larger k retrieves more context but may also "
    "include less relevant chunks."
)


if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun

# Display chat history
for messages in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"**[Source {source['number']}] "
                        f"{source['title']}**"
                    )
                    st.caption(source["source"])
                    st.write(source["content"][:700])


question = st.chat_input(
    "Ask a question about the indexed documents"
)


if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents..."):
                result = answer_question(
                    vector_store=vector_store,
                    llm=chain,
                    question=question,
                    k=k,
                )

            st.markdown(result["answer"])

            if result["sources"]:
                with st.expander("View sources"):
                    for source in result["sources"]:
                        st.markdown(
                            f"**[Source {source['number']}] "
                            f"{source['title']}**"
                        )
                        st.caption(source["source"])
                        st.write(source["content"][:700])

        except Exception as error:
            result = {
                "answer": f"An error occurred: {error}",
                "sources": [],
            }
            st.error(result["answer"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )

