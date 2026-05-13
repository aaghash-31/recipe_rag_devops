import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.rag.rag_service import RagService
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Recipe and Diet Assistant API", version="1.0.0")

# Initialize RAG service
rag_service = RagService()

# Pydantic models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    result: str
    source_documents: list
    retrieval_time: float

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    rag_service.initialize()
    logger.info("Backend startup complete")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "model": "Recipe and Diet Assistant"}

@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a query and return RAG response"""
    try:
        result = rag_service.query(request.query)
        return QueryResponse(
            result=result["result"],
            source_documents=result["source_documents"],
            retrieval_time=result["retrieval_time"],
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
