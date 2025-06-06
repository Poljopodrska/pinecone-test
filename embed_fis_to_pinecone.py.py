# scripts/embed_fis_to_pinecone.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import pinecone

def embed_documents():
    # === Load environment variables ===
    dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
    load_dotenv(dotenv_path)

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")

    # === Load documents ===
    docs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs'))
    print(f"[INFO] Loading HTML files from: {docs_path}")
    loader = DirectoryLoader(docs_path, glob="*.html")
    documents = loader.load()
    print(f"[INFO] Loaded {len(documents)} documents.")

    # === Split documents ===
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"[INFO] Split into {len(chunks)} chunks.")

    # === Embed chunks ===
    embeddings = OpenAIEmbeddings()
    print("[INFO] Embedding chunks...")
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    vectors = embeddings.embed_documents(texts)
    print(f"[INFO] Got {len(vectors)} vectors.")

    # === Init Pinecone ===
    pinecone.init(api_key=pinecone_api_key)

    # Create index if it doesn't exist
    if index_name not in [i.name for i in pinecone.list_indexes()]:
        print(f"[INFO] Creating index '{index_name}'...")
        pinecone.create_index(
            name=index_name,
            dimension=len(vectors[0]),
            metric="cosine"
        )

    index = pinecone.Index(index_name)

    # === Upload in batches ===
    print(f"[INFO] Uploading to Pinecone index: {index_name}...")
    for i in range(0, len(vectors), 100):
        ids = [f"doc-{i+j}" for j in range(len(vectors[i:i+100]))]
        to_upsert = list(zip(ids, vectors[i:i+100], metadatas[i:i+100]))
        index.upsert(vectors=to_upsert)

    print(f"[✅] Upload complete. {len(vectors)} vectors sent to Pinecone index '{index_name}'.")

