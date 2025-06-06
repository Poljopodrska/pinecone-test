import pinecone
import os

# Use direct environment variables (Render supports this)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g. "eu-west-1"

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

print("✅ Pinecone initialized successfully")
print("Indexes:", pinecone.list_indexes())
