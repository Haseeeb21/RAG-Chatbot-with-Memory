from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class QueryRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    query: str = Field(..., description="User's question or query")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "query": "What is the main topic discussed in the documents?"
            }
        }

class RetrievedDocument(BaseModel):
    content: str
    metadata: dict
    relevance_score: float

class QueryResponse(BaseModel):
    answer: str
    retrieved_documents: List[RetrievedDocument]
    conversation_id: str
    timestamp: datetime = Field(default_factory=datetime.now)

class DocumentUploadResponse(BaseModel):
    message: str
    processed_files: List[str]
    total_chunks: int
    status: str

class ConversationHistory(BaseModel):
    user_id: str
    messages: List[dict]
    created_at: datetime
    updated_at: datetime