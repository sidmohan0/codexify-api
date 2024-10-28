# Codexify

Codexify is a powerful open-source FastAPI-based application that enables semantic search and document processing with custom semantic data types. It provides capabilities for document embedding, similarity search, and semantic type management.

[![Docker Image](https://img.shields.io/docker/v/sidmo88/codexify-api/latest?label=Docker%20Image)](https://hub.docker.com/r/sidmo88/codexify-api)

## 🚀 Features

- **Semantic Search**
  - Basic and advanced semantic search capabilities
  - Configurable similarity metrics and filtering
  - Support for multiple embedding models and pooling methods

- **Document Processing**
  - Batch document processing and embedding
  - Support for various file formats
  - Asynchronous processing with job management
  - Document chunking and content retrieval

- **Semantic Data Types**
  - Create and manage custom semantic data types
  - Example-based type definitions
  - Flexible embedding strategies

- **Model Management**
  - Support for multiple LLM models
  - Dynamic model loading and management
  - Model download and status tracking

## 🛠️ Technical Stack

- **Backend Framework**: FastAPI
- **Database**: SQLAlchemy (Async)
- **Vector Search**: FAISS
- **Job Queue**: Redis Queue (RQ)
- **Vector Similarity**: Custom fast vector similarity implementation

## 🔧 Environment Variables

```bash
REDIS_HOST=localhost
REDIS_PORT=6379
DEFAULT_LLM_NAME=Meta-Llama-3-8B-Instruct.Q3_K_S
DEFAULT_EMBEDDING_MODEL_NAME=nomic-embed-text-v1.5.Q6_K
DEFAULT_MULTI_MODAL_MODEL_NAME=llava-llama-3-8b-v1_1-int4
USE_RESOURCE_MONITORING=1
MAX_THOUSANDS_OF_WORDs_FOR_DOCUMENT_EMBEDDING=100
DEFAULT_COMPLETION_TEMPERATURE=0.7
DEFAULT_MAX_COMPLETION_TOKENS=1000
DEFAULT_NUMBER_OF_COMPLETIONS_TO_GENERATE=1
DEFAULT_EMBEDDING_POOLING_METHOD=mean
```

## 📝 API Endpoints

### Document Management
- `POST /documents` - Upload and process documents
- `GET /documents` - Retrieve all documents
- `GET /documents/status/{job_id}` - Check document processing status
- `GET /documents/jobs` - List all document processing jobs
- `POST /documents/delete` - Delete documents
- `GET /document-chunks/{document_hash}` - Get document chunks

### Semantic Search
- `POST /semantic-search/basic` - Perform basic semantic search
- `POST /semantic-search/advanced` - Perform advanced semantic search
- `POST /documents/scan` - Scan documents for semantic analysis

### Semantic Data Types
- `POST /semantic-types/` - Create new semantic data type
- `GET /semantic-types/` - Retrieve all semantic data types

### Model Management
- `GET /models` - List available models
- `POST /models` - Add new model
- `GET /models/status/{job_id}` - Check model download status
- `GET /models/jobs` - List model download jobs

### Similarity Calculation
- `POST /calculate_similarity` - Calculate similarity between strings

## 🚀 Getting Started

1. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Redis server**
   ```bash
   redis-server
   ```

4. **Start the application**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8089
   ```

5. **Start the worker**
   ```bash
   python src/worker.py
   ```

6. **Access the API documentation**
   - Open `http://localhost:8089` in your browser
   - The Swagger UI provides interactive API documentation

## 🔒 Security Notes

- The application includes CORS middleware with configurable origins
- Token-based authentication is supported but optional
- API endpoints include error handling and input validation



## 📞 Support
For any questions or support, please contact hi@datafog.ai 


## Acknowledgements

The design and implementation pattern was inspired by this great project: https://github.com/Dicklesworthstone/swiss_army_llama by Github User Dicklesworthstone.

While the original project provided the foundation for semantic search functionality, this repository includes significant modifications and new features including:
- Environment variable configuration
- Refactored architecture
- Migration from Redis device locks to Redis queue workers
- New endpoints for custom dictionary creation
- New worker implementation for document scanning and semantic search