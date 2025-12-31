import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Optional
import uuid
from app.config import settings

class VectorStore:
    """
    Manages ChromaDB vector database operations.
    
    ChromaDB Features:
    - Local, embedded database (no server needed)
    - Persistent storage
    - Fast similarity search
    - Automatic indexing
    
    Process Flow:
    1. Store: text + embedding + metadata → ChromaDB
    2. Query: input embedding → find similar vectors
    3. Return: most relevant documents with scores
    """
    
    def __init__(self):
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=settings.vector_db_path,
            settings=ChromaSettings(
                anonymized_telemetry=False
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
    
    def add_documents(
        self,
        chunks: List[Dict],
        embeddings: List[List[float]]
    ) -> int:
        """
        Add document chunks with embeddings to vector store.
        
        Args:
            chunks: List of document chunks with content and metadata
            embeddings: Corresponding embeddings for each chunk
            
        Returns:
            Number of documents added
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        # Prepare data for ChromaDB
        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [chunk['content'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"Added {len(chunks)} documents to vector store")
        return len(chunks)
    
    def similarity_search(
        self,
        query_embedding: List[float],
        n_results: int = 5
    ) -> List[Dict]:
        """
        Search for similar documents using vector similarity.
        
        Similarity Metrics:
        - Cosine similarity: measures angle between vectors
        - Range: -1 (opposite) to 1 (identical)
        - Higher score = more similar
        
        Args:
            query_embedding: Embedding vector of the query
            n_results: Number of results to return
            
        Returns:
            List of documents with content, metadata, and relevance scores
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Format results
        documents = []
        for i in range(len(results['ids'][0])):
            documents.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'relevance_score': 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        return documents
    
    def clear_collection(self):
        """Delete all documents from the collection."""
        self.client.delete_collection(name=settings.collection_name)
        self.collection = self.client.create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("Collection cleared")
    
    def get_collection_count(self) -> int:
        """Get total number of documents in collection."""
        return self.collection.count()
    
    def delete_by_source(self, source: str):
        """Delete documents from a specific source file."""
        # Get all items with matching source
        results = self.collection.get(
            where={"source": source}
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            print(f"Deleted {len(results['ids'])} documents from {source}")