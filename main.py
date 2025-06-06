from fastapi import FastAPI
import uvicorn
import traceback

# Import your existing embedding function
from embed_fis_to_pinecone import embed_documents  # Adjust if your function name is different

app = FastAPI()

@app.get("/")
def root():
    return {"status": "🟢 Online - SA Ready"}

@app.get("/embed-fis")
def run_embedding():
    try:
        embed_documents()
        return {"status": "✅ Embedding complete!"}
    except Exception as e:
        return {
            "status": "❌ Failed",
            "error": str(e),
            "trace": traceback.format_exc()
        }

# Optional: For local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
