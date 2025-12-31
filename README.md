# RAG-Chatbot-with-Memory

A production-ready Retrieval-Augmented Generation (RAG) chatbot with per-user conversation memory. Query your documents (PDF, TXT, DOCX) using natural language with ChromaDB vector search and OpenAI embeddings.

## ✨ Features

- 📚 **Document Processing**: Support for PDF, TXT, and DOCX files
- 🔍 **Semantic Search**: ChromaDB vector database with cosine similarity
- 🧠 **User Memory**: Per-user conversation history with sliding window
- 🚀 **Fast API**: RESTful API with FastAPI and automatic documentation
- 🔐 **Local Storage**: Documents and conversations stored locally
- 💬 **Context-Aware**: Maintains conversation flow and understands follow-up questions
- 📊 **Source Attribution**: Cites source documents in responses

## 🏗️ Architecture

```
User Query → Embedding → Vector Search → Context Building → LLM → Response
                ↓                            ↓
          ChromaDB (Local)          Conversation Memory (Local)
```

**Tech Stack:**
- **Backend**: FastAPI
- **Vector DB**: ChromaDB (local, embedded)
- **LLM**: OpenAI GPT-4, etc.
- **Embeddings**: OpenAI text-embedding-3-small
- **Document Processing**: LangChain, PyPDF, python-docx

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- OpenAI API key 

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/haseeeb21/rag-chatbot-with-memory.git
cd rag-chatbot-with-memory
```

2. **Create virtual environment**
```bash
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=sk-your-api-key-here
```

5. **Add your documents**
```bash
# Place your PDF, TXT, DOCX files in:
data/documents/
```

6. **Run the server**
```bash
python -m app.main
```

Server will start at: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

## 📖 Usage

### 1. Index Your Documents

First, index your documents to create the knowledge base:

```bash
curl -X POST http://localhost:8000/index
```

**Response:**
```json
{
  "message": "Documents indexed successfully",
  "processed_files": ["document1.pdf", "document2.txt"],
  "total_chunks": 45,
  "status": "completed"
}
```

### 2. Query the System

Ask questions about your documents:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "ali",
    "query": "What are the main topics in the documents?"
  }'
```

**Response:**
```json
{
  "answer": "Based on the documents, the main topics include...",
  "retrieved_documents": [
    {
      "content": "Relevant excerpt...",
      "metadata": {"filename": "document1.pdf"},
      "relevance_score": 0.89
    }
  ],
  "conversation_id": "ali",
  "timestamp": "2024-12-31T10:30:00"
}
```

### 3. Follow-up Questions

The system remembers your conversation:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "ali",
    "query": "Can you explain more about that?"
  }'
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/query` | Ask a question |
| `POST` | `/index` | Index documents |
| `GET` | `/history/{user_id}` | Get conversation history |
| `DELETE` | `/history/{user_id}` | Clear user history |
| `GET` | `/stats` | System statistics |
| `DELETE` | `/clear-db` | Clear vector database |

### Full API Documentation

Visit `http://localhost:8000/docs` for interactive Swagger UI documentation.

## 🧪 Testing with Postman

1. **Import Collection**: Use the provided `/openapi.json` in the docs page.
2. **Set Base URL**: `http://localhost:8000`
3. **Test Sequence**:
   - POST `/index` - Index documents
   - POST `/query` - Ask first question
   - POST `/query` - Ask follow-up
   - GET `/history/{user_id}` - Check memory

## 📁 Project Structure

```
rag-chatbot-with-memory/
├── app/
│   ├── services/          # Core business logic
│   │   ├── document_processor.py
│   │   ├── embedding_service.py
│   │   ├── vector_store.py
│   │   ├── rag_service.py
│   │   └── memory_service.py
│   ├── main.py           # FastAPI application
│   ├── models.py         # Pydantic models
│   └── config.py         # Configuration
├── data/
│   └── documents/        # Your documents (PDF, TXT, DOCX)
├── storage/
│   ├── chroma_db/        # Vector database
│   └── conversations/    # User conversation history
└── requirements.txt
```

## 🔧 Configuration

All settings can be configured in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | Your OpenAI API key |
| `EMBEDDING_MODEL` | text-embedding-3-small | Embedding model |
| `LLM_MODEL` | gpt-4-turbo-preview | Language model |
| `CHUNK_SIZE` | 1000 | Document chunk size |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `MAX_CONVERSATION_HISTORY` | 10 | Messages to keep in memory |

## 📊 How It Works

### 1. Document Indexing Pipeline
```
Documents → Text Extraction → Chunking → Embeddings → Vector Store
```

### 2. Query Processing Pipeline
```
Query → Embedding → Similarity Search → Context Building → LLM → Response
          ↓                                   ↓
    ChromaDB Search                  Conversation History
```

### 3. Memory Management
- Per-user conversation storage
- Sliding window (keeps last N messages)
- Persistent to disk
- Included in context for follow-ups

## 🎯 Use Cases

- **Personal Knowledge Base**: Query your research papers, notes, and documents
- **Document Q&A**: Get answers from technical documentation
- **Customer Support**: Build a chatbot for your product docs
- **Research Assistant**: Analyze multiple documents and extract insights
- **Legal/Medical Document Analysis**: Query domain-specific documents

## 🔒 Privacy & Security

- ✅ Documents stored locally
- ✅ Conversation history stored locally
- ✅ Only embeddings sent to OpenAI (not full documents)
- ⚠️ API calls to OpenAI (queries and responses)
- 🔐 No data shared between users

## 🚧 Limitations

- Requires OpenAI API (not fully offline)
- Limited by OpenAI rate limits
- Vector database size depends on document count
- Context window limited by LLM


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [LangChain](https://www.langchain.com/) - LLM framework
- [OpenAI](https://openai.com/) - LLM and embeddings


---

⭐ If you find this project useful, please consider giving it a star on GitHub!
