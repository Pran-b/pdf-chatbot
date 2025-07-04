import os
import boto3
from langchain.vectorstores import FAISS
from langchain.embeddings import BedrockEmbeddings
from langchain.embeddings import OllamaEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.llms.ollama import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


def get_vectorstore(pages: list[str], provider: str, aws_region: str = None, aws_access_key: str = None, aws_secret_key: str = None) -> FAISS:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = [Document(page_content=p) for p in pages]
    splits = text_splitter.split_documents(docs)

    if provider == "ollama":
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    else:
        if aws_region and aws_access_key and aws_secret_key:
            boto3.setup_default_session(
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
            )
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")

    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore


def get_qa_chain(vectorstore: FAISS, provider: str):
    if provider == "ollama":
        llm = Ollama(model="mistral")
    else:
        llm = Bedrock(model_id="anthropic.claude-3-sonnet-20240229")

    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
