import streamlit as st
from qa_engine import get_vectorstore, get_qa_chain
from utils import extract_pages_from_pdf
import tempfile
import subprocess

# --- Streamlit App ---
st.set_page_config(page_title="PDF Q&A", page_icon="📄")
st.title("📄 Ask Questions About Your PDF (Powered by AWS Bedrock or Ollama)")

provider = st.selectbox("Choose LLM Provider", ["bedrock", "ollama"], key="llm_provider")

if provider == "bedrock":
    aws_region = st.text_input("AWS Region", value="us-east-1", key="aws_region")
    aws_access_key = st.text_input("AWS Access Key ID", type="password", key="aws_access")
    aws_secret_key = st.text_input("AWS Secret Access Key", type="password", key="aws_secret")
    ollama_model = None
else:
    # Auto-detect Ollama models
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().splitlines()[1:]  # skip header
        model_names = [line.split()[0] for line in lines if line.strip()]
        model_error = None
    except Exception as e:
        model_names = ["mistral", "llama2", "gemma"]
        model_error = str(e)

    model_names.append("other")
    ollama_model = st.selectbox("Choose Ollama Model", model_names, key="ollama_model")
    if ollama_model == "other":
        ollama_model = st.text_input("Enter custom Ollama model name", key="ollama_custom_model")

    aws_region = aws_access_key = aws_secret_key = None

    if model_error:
        st.warning(f"⚠️ Could not auto-detect Ollama models: {model_error}")

pdf_file = st.file_uploader("Upload your PDF", type="pdf", key="pdf_uploader")

if pdf_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    pages = extract_pages_from_pdf(tmp_path)
    vs = get_vectorstore(pages, provider, aws_region, aws_access_key, aws_secret_key)
    model_id = provider if provider == "bedrock" else f"ollama:{ollama_model}"
    qa = get_qa_chain(vs, model_id)

    question = st.text_input("Ask a question from the PDF")
    if question:
        result = qa.run(question)
        st.write(result)

