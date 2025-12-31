from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os

from app.models import (
    QueryRequest,
    QueryResponse,
    RetrievedDocument,
    DocumentUploadResponse
)
from app.services.rag_service import RAGService
from app.config import settings

# Initialize FastAPI app
app = FastAPI(
    title="RAG AI Agent API",
    description="Retrieval-Augmented Generation system with user memory",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG service
rag_service = RAGService()

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    print("="*60)
    print("RAG AI Agent Starting Up...")
    print("="*60)
    
    # Create necessary directories
    os.makedirs("./data/documents", exist_ok=True)
    os.makedirs(settings.vector_db_path, exist_ok=True)
    os.makedirs(settings.memory_path, exist_ok=True)
    
    print(f"✓ Directories created")
    print(f"✓ Vector DB path: {settings.vector_db_path}")
    print(f"✓ Memory path: {settings.memory_path}")
    print(f"✓ Using model: {settings.llm_model}")
    
    # Check if documents are indexed
    count = rag_service.vector_store.get_collection_count()
    print(f"✓ Current indexed documents: {count}")
    
    if count == 0:
        print("\n⚠️  No documents indexed yet!")
        print("   Use POST /index to index documents from ./data/documents/")
    
    print("="*60)
    print("API ready! Visit http://localhost:8000/docs for Swagger UI")
    print("="*60 + "\n")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "message": "RAG AI Agent API",
        "timestamp": datetime.now().isoformat(),
        "indexed_documents": rag_service.vector_store.get_collection_count()
    }

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Process a user query using RAG.
    
    Flow:
    1. Receive user query
    2. Embed query
    3. Retrieve relevant documents
    4. Generate response with LLM
    5. Save to user's conversation history
    
    Example Request:
    ```json
    {
        "user_id": "user123",
        "query": "What is discussed in the documents?"
    }
    ```
    """
    try:
        # Process query
        answer, retrieved_docs = rag_service.query(
            user_id=request.user_id,
            query=request.query
        )
        
        # Format response
        response = QueryResponse(
            answer=answer,
            retrieved_documents=[
                RetrievedDocument(
                    content=doc['content'][:200] + "...",  # Truncate for response
                    metadata=doc['metadata'],
                    relevance_score=doc['relevance_score']
                )
                for doc in retrieved_docs
            ],
            conversation_id=request.user_id,
            timestamp=datetime.now()
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index", response_model=DocumentUploadResponse)
async def index_documents(background_tasks: BackgroundTasks):
    """
    Index all documents from ./data/documents/ directory.
    
    Process:
    1. Load documents (PDF, TXT, DOCX)
    2. Split into chunks
    3. Generate embeddings
    4. Store in ChromaDB
    
    This endpoint processes documents in the background.
    """
    try:
        documents_dir = "./data/documents"
        
        if not os.path.exists(documents_dir):
            raise HTTPException(
                status_code=400,
                detail=f"Documents directory not found: {documents_dir}"
            )
        
        # Get list of files
        files = [f for f in os.listdir(documents_dir) 
                if f.endswith(('.txt', '.pdf', '.docx'))]
        
        if not files:
            raise HTTPException(
                status_code=400,
                detail="No documents found in ./data/documents/"
            )
        
        # Index documents
        total_chunks = rag_service.index_documents(documents_dir)
        
        return DocumentUploadResponse(
            message="Documents indexed successfully",
            processed_files=files,
            total_chunks=total_chunks,
            status="completed"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{user_id}")
async def get_conversation_history(user_id: str):
    """
    Get conversation history for a specific user.
    
    Returns all stored messages for the user.
    """
    try:
        history = rag_service.memory_service.get_history(user_id)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/history/{user_id}")
async def clear_conversation_history(user_id: str):
    """
    Clear conversation history for a specific user.
    """
    try:
        rag_service.memory_service.clear_history(user_id)
        return {"message": f"History cleared for user: {user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """
    Get system statistics.
    """
    try:
        return {
            "total_documents": rag_service.vector_store.get_collection_count(),
            "total_users": len(rag_service.memory_service.get_all_users()),
            "model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear-db")
async def clear_database():
    """
    Clear all documents from vector database.
    
    ⚠️ WARNING: This will delete all indexed documents!
    """
    try:
        rag_service.vector_store.clear_collection()
        return {"message": "Vector database cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)