from typing import List, Dict, Tuple
from openai import OpenAI
from app.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.memory_service import MemoryService

class RAGService:
    """
    Core RAG (Retrieval-Augmented Generation) Service.
    
    RAG Pipeline:
    1. User Query → Generate Embedding
    2. Embedding → Search Vector DB for Similar Documents
    3. Retrieved Docs + Query + Memory → Build Context
    4. Context → LLM → Generate Answer
    5. Store Interaction in Memory
    
    Benefits of RAG:
    - Grounds responses in your documents
    - Reduces hallucinations
    - Provides source attribution
    - Up-to-date information from your data
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.memory_service = MemoryService()
        self.model = settings.llm_model
        self.temperature = settings.temperature
        self.max_tokens = settings.max_tokens
    
    def query(
        self,
        user_id: str,
        query: str,
        n_results: int = 5
    ) -> Tuple[str, List[Dict]]:
        """
        Process a user query using RAG.
        
        Complete Flow:
        1. Generate query embedding
        2. Retrieve relevant documents
        3. Get conversation context
        4. Build prompt with context
        5. Generate response with LLM
        6. Save to memory
        
        Args:
            user_id: Unique user identifier
            query: User's question
            n_results: Number of documents to retrieve
            
        Returns:
            Tuple of (answer, retrieved_documents)
        """
        print(f"\n{'='*60}")
        print(f"Processing Query for User: {user_id}")
        print(f"Query: {query}")
        print(f"{'='*60}\n")
        
        # Step 1: Generate embedding for the query
        print("Step 1: Generating query embedding...")
        query_embedding = self.embedding_service.generate_embedding(query)
        print(f"✓ Generated embedding with {len(query_embedding)} dimensions\n")
        
        # Step 2: Retrieve relevant documents
        print(f"Step 2: Searching vector database for top {n_results} matches...")
        retrieved_docs = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            n_results=n_results
        )
        print(f"✓ Retrieved {len(retrieved_docs)} documents")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"  Doc {i}: {doc['metadata'].get('filename', 'unknown')} "
                  f"(score: {doc['relevance_score']:.3f})")
        print()
        
        # Step 3: Get conversation history
        print("Step 3: Loading conversation history...")
        conversation_context = self.memory_service.get_conversation_context(
            user_id=user_id,
            n_messages=3
        )
        print(f"✓ Loaded conversation context\n")
        
        # Step 4: Build context from retrieved documents
        print("Step 4: Building context from retrieved documents...")
        context = self._build_context(retrieved_docs)
        print(f"✓ Built context with {len(context)} characters\n")
        
        # Step 5: Generate response
        print("Step 5: Generating response from LLM...")
        answer = self._generate_response(
            query=query,
            context=context,
            conversation_history=conversation_context
        )
        print(f"✓ Generated response\n")
        
        # Step 6: Save to memory
        print("Step 6: Saving interaction to memory...")
        self.memory_service.add_interaction(
            user_id=user_id,
            user_message=query,
            assistant_message=answer,
            retrieved_context=retrieved_docs
        )
        print(f"✓ Saved to memory\n")
        
        print(f"{'='*60}")
        print("Query processing complete!")
        print(f"{'='*60}\n")
        
        return answer, retrieved_docs
    
    def _build_context(self, documents: List[Dict]) -> str:
        """
        Build context string from retrieved documents.
        
        Format documents with source attribution for the LLM.
        """
        context = "Relevant information from documents:\n\n"
        
        for i, doc in enumerate(documents, 1):
            source = doc['metadata'].get('filename', 'unknown')
            content = doc['content']
            score = doc['relevance_score']
            
            context += f"[Document {i}] (Source: {source}, Relevance: {score:.2f})\n"
            context += f"{content}\n\n"
        
        return context
    
    def _generate_response(
        self,
        query: str,
        context: str,
        conversation_history: str
    ) -> str:
        """
        Generate response using OpenAI with RAG context.
        
        Prompt Engineering:
        - System message: defines AI behavior
        - Context: retrieved documents
        - History: previous conversation
        - Query: current question
        """
        # Build the prompt
        system_message = """You are a helpful AI assistant with access to a knowledge base. 
Your task is to answer questions based on the provided context and conversation history.

Guidelines:
1. Answer based primarily on the provided context
2. If the context doesn't contain relevant information, say so clearly
3. Cite sources when possible (mention document names)
4. Maintain conversation continuity using the history
5. Be concise but complete
6. If asked about previous messages, refer to the conversation history"""
        
        user_message = f"""{conversation_history}

Context from knowledge base:
{context}

Current question: {query}

Please provide a helpful answer based on the context above."""
        
        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content
    
    def index_documents(self, directory: str) -> int:
        """
        Index all documents from a directory into the vector store.
        
        Complete Indexing Pipeline:
        1. Load documents from directory
        2. Chunk documents
        3. Generate embeddings
        4. Store in vector database
        
        Args:
            directory: Path to documents directory
            
        Returns:
            Number of chunks indexed
        """
        from app.services.document_processor import DocumentProcessor
        
        print(f"\n{'='*60}")
        print(f"Starting Document Indexing Pipeline")
        print(f"Directory: {directory}")
        print(f"{'='*60}\n")
        
        processor = DocumentProcessor()
        
        # Step 1: Load documents
        print("Step 1: Loading documents...")
        documents = processor.load_documents(directory)
        print(f"✓ Loaded {len(documents)} documents\n")
        
        # Step 2: Chunk documents
        print("Step 2: Chunking documents...")
        chunks = processor.chunk_documents(documents)
        print(f"✓ Created {len(chunks)} chunks\n")
        
        # Step 3: Generate embeddings
        print("Step 3: Generating embeddings...")
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embedding_service.generate_embeddings_batch(texts)
        print(f"✓ Generated {len(embeddings)} embeddings\n")
        
        # Step 4: Store in vector database
        print("Step 4: Storing in vector database...")
        count = self.vector_store.add_documents(chunks, embeddings)
        print(f"✓ Indexed {count} chunks\n")
        
        print(f"{'='*60}")
        print(f"Indexing Complete! Total chunks: {count}")
        print(f"{'='*60}\n")
        
        return count