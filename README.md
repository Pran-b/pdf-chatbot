# pdf-chatbot
# 📘 Project: PDF Q&A Bot using AWS Bedrock + LangChain + Ollama

"""
Ask questions over PDFs using Claude 3 (via AWS Bedrock) or local Ollama models with automatic chunking and semantic search.
"""

# 🧱 Tech Stack:
# - AWS Bedrock (Claude 3 Sonnet or Titan)
# - Ollama (Mistral, LLaMA, etc.) for local fallback
# - LangChain for RAG flow
# - PyPDF2 for PDF parsing
# - FAISS for vector store (in-memory)
# - Streamlit for UI (or optional CLI)

# ✅ Repo Structure:
# - app.py                # Main Streamlit app
# - qa_engine.py          # Embedding, vector store, and model interface
# - utils.py              # PDF parsing and preprocessing
# - requirements.txt      # Dependencies
# - README.md             # Project overview & instructions
