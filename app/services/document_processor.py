import os
from typing import List, Dict
from pathlib import Path
from pypdf import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config import settings

class DocumentProcessor:
    """
    Handles document loading, text extraction, and chunking.
    
    Process Flow:
    1. Load documents from directory (PDF, TXT, DOCX)
    2. Extract text content
    3. Split into chunks with overlap
    4. Return chunks with metadata
    """
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_documents(self, directory: str) -> List[Dict]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory: Path to documents folder
            
        Returns:
            List of dictionaries with 'content' and 'metadata'
        """
        documents = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Supported file extensions
        supported_extensions = {'.txt', '.pdf', '.docx'}
        
        for file_path in directory_path.rglob('*'):
            if file_path.suffix.lower() in supported_extensions:
                print(f"Processing: {file_path.name}")
                
                try:
                    content = self._extract_text(file_path)
                    if content.strip():
                        documents.append({
                            'content': content,
                            'metadata': {
                                'source': str(file_path),
                                'filename': file_path.name,
                                'file_type': file_path.suffix
                            }
                        })
                except Exception as e:
                    print(f"Error processing {file_path.name}: {str(e)}")
        
        return documents
    
    def _extract_text(self, file_path: Path) -> str:
        """Extract text based on file type."""
        extension = file_path.suffix.lower()
        
        if extension == '.txt':
            return self._read_txt(file_path)
        elif extension == '.pdf':
            return self._read_pdf(file_path)
        elif extension == '.docx':
            return self._read_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    def _read_txt(self, file_path: Path) -> str:
        """Read text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _read_pdf(self, file_path: Path) -> str:
        """Read PDF file."""
        reader = PdfReader(str(file_path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def _read_docx(self, file_path: Path) -> str:
        """Read DOCX file."""
        doc = Document(str(file_path))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Split documents into smaller chunks.
        
        Chunking Strategy:
        - Uses RecursiveCharacterTextSplitter
        - Maintains context with overlap
        - Preserves metadata for each chunk
        
        Args:
            documents: List of documents with content and metadata
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        for doc in documents:
            # Split the text into chunks
            text_chunks = self.text_splitter.split_text(doc['content'])
            
            # Create chunk objects with metadata
            for i, chunk in enumerate(text_chunks):
                chunk_metadata = doc['metadata'].copy()
                chunk_metadata['chunk_id'] = i
                chunk_metadata['total_chunks'] = len(text_chunks)
                
                chunks.append({
                    'content': chunk,
                    'metadata': chunk_metadata
                })
        
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
