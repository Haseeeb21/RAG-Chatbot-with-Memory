from typing import List
from openai import OpenAI
from app.config import settings

class EmbeddingService:
    """
    Handles text embedding generation using OpenAI.
    
    Process:
    1. Takes text input
    2. Calls OpenAI Embeddings API
    3. Returns vector representation
    
    Why Embeddings?
    - Convert text to numerical vectors
    - Enable semantic similarity search
    - Capture meaning, not just keywords
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        # Clean and prepare text
        text = text.replace("\n", " ").strip()
        
        if not text:
            raise ValueError("Text cannot be empty")
        
        # Call OpenAI API
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        
        return response.data[0].embedding
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        More efficient for processing multiple documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Clean texts
        cleaned_texts = [text.replace("\n", " ").strip() for text in texts]
        
        # Batch API call
        response = self.client.embeddings.create(
            input=cleaned_texts,
            model=self.model
        )
        
        # Extract embeddings in order
        embeddings = [item.embedding for item in response.data]
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings from this model.
        
        text-embedding-3-small: 1536 dimensions
        text-embedding-3-large: 3072 dimensions
        """
        # Generate a test embedding to determine dimension
        test_embedding = self.generate_embedding("test")
        return len(test_embedding)