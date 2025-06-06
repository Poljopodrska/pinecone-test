# scripts/embed_runner.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone

def run_fis_embedding():
    dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
    load_dotenv(dotenv_path)

    index_name = os.getenv("PINECONE_INDEX_NAME")
    docs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs'))
    print(f"[INFO] Loading HTML files from: {docs_path}")

    loader = DirectoryLoader(docs_path, glob="*.html")
    documents = loader.load()
    print(f"[INFO] Loaded {len(documents)} documents.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"[INFO] Split into {len(chunks)} chunks.")

    embeddings = OpenAIEmbeddings()
    print("[INFO] Embedding chunks...")
    vectorstore = LangchainPinecone.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        namespace=""
    )
    print(f"[✅] Upload complete. {len(chunks)} chunks embedded into Pinecone index: {index_name}")
