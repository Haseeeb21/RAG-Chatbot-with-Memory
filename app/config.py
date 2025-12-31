from pydantic_settings import BaseSettings
from typing import Optional
import os

# Disable ChromaDB telemetry to prevent capture() errors
# os.environ["CHROMA_TELEMETRY_IMPL"] = "none"

class Settings(BaseSettings):
    # OpenAI Configuration
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4-turbo-preview"
    temperature: float = 0.7
    max_tokens: int = 1000
    
    # Document Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Vector Store
    vector_db_path: str = "./storage/chroma_db"
    collection_name: str = "rag_documents"
    
    # Memory
    memory_path: str = "./storage/conversations"
    max_conversation_history: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()