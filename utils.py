import PyPDF2

def extract_pages_from_pdf(file_path: str) -> list[str]:
    reader = PyPDF2.PdfReader(file_path)
    return [page.extract_text() for page in reader.pages if page.extract_text()]


# --- app.py ---

import streamlit as st
from qa_engine import get_vectorstore, get_qa_chain
from utils import extract_pages_from_pdf
import tempfile

st.title("📄 Ask Questions About Your PDF (Powered by AWS Bedrock or Ollama)")

provider = st.selectbox("Choose LLM Provider", ["bedrock", "ollama"])
pdf_file = st.file_uploader("Upload your PDF", type="pdf")

if pdf_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    pages = extract_pages_from_pdf(tmp_path)
    vs = get_vectorstore(pages, provider)
    qa = get_qa_chain(vs, provider)

    question = st.text_input("Ask a question from the PDF")
    if question:
        result = qa.run(question)
        st.write(result)
