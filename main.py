from fastapi import FastAPI
import traceback

from embed_fis_to_pinecone import embed_documents  # Make sure this exists and works

app = FastAPI()

@app.get("/")
def root():
    return {"status": "🟢 Online - SA Ready"}

@app.post("/embed")
def run_embedding():
    try:
        embed_documents()
        return {"status": "✅ Embedding complete!"}
    except Exception as e:
        return {
            "status": "❌ Embedding failed",
            "error": str(e),
            "trace": traceback.format_exc()
        }
