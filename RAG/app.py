import os
from pathlib import Path

import streamlit as st

from rag.generation import generate
from rag.store import get_collection, retrieve


ROOT = Path(__file__).resolve().parent
DB_DIR = ROOT / "chroma_db"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_dotenv(ROOT / ".env")

st.set_page_config(page_title="Document RAG Chatbot", page_icon=":page_facing_up:", layout="wide")
st.title("Document RAG Chatbot")
st.caption("Ask questions about the documents stored in the documents folder.")

with st.sidebar:
    st.header("Vector database")
    upload_dir = ROOT / "documents" / "uploads"

    uploaded_pdf = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Upload a text-based PDF. The file replaces the current document collection.",
    )

    if uploaded_pdf is not None:
        upload_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = upload_dir / uploaded_pdf.name
        pdf_path.write_bytes(uploaded_pdf.getbuffer())

        upload_key = f"{uploaded_pdf.name}:{uploaded_pdf.size}"
        if st.session_state.get("uploaded_document_key") != upload_key:
            from ingest import ingest

            try:
                with st.spinner(f"Ingesting {uploaded_pdf.name}..."):
                    count = ingest(pdf_path)
                st.session_state["uploaded_document_key"] = upload_key
                st.session_state["current_document"] = uploaded_pdf.name
                st.session_state.messages = []
                st.success(f"{uploaded_pdf.name} ingested ({count} chunks).")
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                st.error(str(exc))
        elif st.session_state.get("current_document"):
            st.caption(f"Active document: {st.session_state['current_document']}")

    if st.session_state.get("current_document"):
        if st.button("Re-ingest active PDF", use_container_width=True):
            from ingest import ingest

            active_pdf = upload_dir / st.session_state["current_document"]
            if active_pdf.exists():
                count = ingest(active_pdf)
                st.success(f"Re-ingested {count} chunks.")

    if DB_DIR.exists():
        st.success("Chroma database is available.")
    else:
        st.info("Run ingestion before asking questions.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your documents...")
if question:
    if DB_DIR.exists():
        try:
            test_collection = get_collection(DB_DIR)
            if test_collection.count() == 0:
                st.error("Upload and ingest a PDF before asking questions.")
                st.stop()
        except Exception as exc:
            st.error(f"Could not open the vector database: {exc}")
            st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if not DB_DIR.exists():
        answer = "Please ingest the documents first using the sidebar."
        sources = []
    else:
        try:
            collection = get_collection(DB_DIR)
            active_document = st.session_state.get("current_document")
            hits = retrieve(
                collection,
                question,
                top_k=3,
                source=active_document,
            )
            answer = generate(question, hits)
            sources = sorted({hit["source"] for hit in hits})
        except Exception as exc:
            answer = f"Unable to answer: {exc}"
            sources = []

    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            st.caption("Sources: " + ", ".join(sources))

    st.session_state.messages.append({"role": "assistant", "content": answer})
