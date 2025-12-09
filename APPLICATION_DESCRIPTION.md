# News Research Tool - Complete Application Description

## Project Overview
The **News Research Tool** is an intelligent AI-powered web application that allows users to analyze and extract insights from news articles using advanced natural language processing and vector search technology. It combines LangChain, OpenAI's GPT models, and FAISS vector database to enable semantic search and question-answering capabilities across multiple news sources.

---

## Core Application Architecture

### Technology Stack
- **Frontend**: Streamlit (Python-based web framework)
- **LLM & Embeddings**: OpenAI API (GPT models + embedding models)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Text Processing**: LangChain + RecursiveCharacterTextSplitter
- **HTTP Requests**: Python requests library
- **Environment Management**: python-dotenv

---

## Application Workflow (Step-by-Step)

### Phase 1: User Input
- Users input up to 3 news article URLs through Streamlit's sidebar interface
- Click "Process URLs" button to initiate the pipeline

### Phase 2: Document Loading
- Custom `_load_documents_from_urls()` function fetches HTML content from provided URLs
- Includes proper User-Agent headers to mimic browser requests
- HTML is parsed using `_HTMLTextExtractor` class to extract plain text
- Creates `Document` objects with page content and source metadata
- Handles errors gracefully (timeouts, failed requests, empty content)

### Phase 3: Text Splitting
- `RecursiveCharacterTextSplitter` breaks documents into manageable chunks
- **Configuration**:
  - Chunk size: 1000 characters
  - Chunk overlap: 200 characters (for context continuity)
  - Split hierarchy: paragraphs → sentences → words
- This ensures semantic coherence within each chunk

### Phase 4: Embedding & Vector Storage
- Uses OpenAI's embedding model to convert text chunks into high-dimensional vectors
- Creates a FAISS vector index from embeddings
- Saves the index locally in `faiss_store_openai/` directory
- FAISS enables fast similarity search across documents

### Phase 5: Question Processing & Retrieval
- User enters a question through the main interface
- The question is converted to an embedding using the same OpenAI model
- FAISS retriever searches the vector store and returns top 4 most similar chunks
- Retrieved chunks become the "context" for the LLM

### Phase 6: Response Generation
- `SimpleRetrievalQAWithSourcesChain` custom class handles the Q&A pipeline
- Combines retrieved context with the user's question
- Sends prompt to OpenAI's ChatGPT with system instruction to:
  - Use ONLY provided context (prevents hallucination)
  - Cite sources in parentheses
  - Respond in 4-6 sentences
- Returns structured response with answer text and source URLs

---

## Key Components

### 1. SimpleRetrievalQAWithSourcesChain Class
**Purpose**: Custom retrieval QA chain compatible with latest LangChain versions

**Key Methods**:
- `invoke()`: Main method that executes the Q&A pipeline
- `_format_docs()`: Formats retrieved documents with source labels
- `_collect_sources()`: Extracts unique source URLs from retrieved documents
- `from_chain_type()`: Factory method for initialization

**Key Features**:
- Accepts both string and dictionary inputs for flexibility
- Automatically formats context from retrieved documents
- Maintains source attribution throughout the pipeline
- Compatible with different retriever types (vectorstore and custom retrievers)

### 2. _HTMLTextExtractor Class
- Lightweight HTML parser (avoids heavy dependencies like BeautifulSoup)
- Strips HTML markup and extracts clean text content
- Preserves text structure through newline characters
- Extends Python's native `HTMLParser` for minimal dependencies

### 3. _load_documents_from_urls() Function
- Fetches content from provided URLs using the requests library
- Implements proper HTTP headers to avoid blocking
- Error handling for network failures and timeouts
- Returns list of `Document` objects with metadata

### 4. LLM Configuration
```
Temperature: 0.9 (allows creative but coherent responses)
Max Tokens: 500 (ensures concise answers)
Model: gpt-3.5-turbo or gpt-4 (from OpenAI)
Embeddings: OpenAI embedding model (default: text-embedding-ada-002)
```

---

## Data Flow Diagram

```
User URLs (Sidebar Input)
    ↓
[Process URLs Button Clicked]
    ↓
HTML Fetching & Parsing (_load_documents_from_urls)
    ↓
Plain Text Extraction (_HTMLTextExtractor)
    ↓
Document Creation (LangChain Document objects)
    ↓
Text Splitting (RecursiveCharacterTextSplitter)
    ↓
Chunk Size: 1000 chars, Overlap: 200 chars
    ↓
OpenAI Embeddings (Vector conversion)
    ↓
FAISS Vector Index Creation
    ↓
Save Index Locally (faiss_store_openai/)
    ↓
[User enters Question]
    ↓
Question Embedding
    ↓
Vector Similarity Search (FAISS retriever)
    ↓
Top 4 Most Similar Chunks Retrieved
    ↓
Context Formatting (_format_docs)
    ↓
Prompt Construction (ChatPromptTemplate)
    ↓
ChatGPT LLM Processing (SimpleRetrievalQAWithSourcesChain)
    ↓
Answer Generation with Source Citations
    ↓
Display Results (Streamlit UI)
```

---

## Key Features & Benefits

| Feature | Implementation | Benefit |
|---------|-----------------|---------|
| **Multi-Source Analysis** | Process 3 URLs simultaneously | Compare multiple perspectives on a topic |
| **Semantic Search** | FAISS vector similarity matching | Find relevant info even without exact keyword match |
| **Source Attribution** | Automatic citation from metadata | Verify information and trace back to sources |
| **Error Handling** | Graceful failures with user warnings | Robust application that handles network issues |
| **Persistent Storage** | FAISS index saved for reuse | Avoid re-processing and embedding same articles |
| **User-Friendly UI** | Streamlit dashboard with clear workflow | No technical knowledge required |
| **Context-Limited Responses** | Prevents AI hallucination | Ensures factual accuracy based on sources |
| **Chunk Overlap** | 200-character overlap | Preserve context across chunk boundaries |

---

## Unique Design Decisions

1. **Custom Retrieval Chain**: Instead of using LangChain's built-in chains, implements a minimal custom class compatible with latest packages
   - Reason: Ensures compatibility without being tied to specific LangChain versions

2. **Lightweight HTML Parsing**: Uses native Python `HTMLParser` instead of external libraries
   - Reason: Reduces dependencies and improves performance

3. **Chunk Overlap Strategy**: 200-character overlap between chunks
   - Reason: Ensures information split across chunk boundaries isn't lost

4. **Local Vector Storage**: FAISS index persisted for efficient reuse without re-embedding
   - Reason: Saves API costs and processing time

5. **Recursive Text Splitter**: Hierarchical splitting preserves semantic meaning better than simple chunking
   - Reason: Maintains semantic coherence for better retrieval

6. **Top-K Retrieval (k=4)**: Retrieves 4 most relevant chunks per question
   - Reason: Balances between comprehensive context and prompt token limits

---

## Configuration Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Temperature | 0.9 | Controls response creativity vs consistency |
| Max Tokens | 500 | Limits response length |
| Chunk Size | 1000 characters | Size of text segments for processing |
| Chunk Overlap | 200 characters | Context preservation between chunks |
| Retrieval Count (k) | 4 | Number of similar chunks to retrieve |
| Max URLs | 3 | Number of articles to process simultaneously |
| HTTP Timeout | 30 seconds | URL fetching timeout |

---

## Error Handling Strategy

The application implements comprehensive error handling:

1. **URL Loading Errors**:
   - Network timeouts (30-second timeout)
   - HTTP errors (non-200 responses)
   - Empty/unreadable content
   - User warning messages for each failure

2. **Vector Store Errors**:
   - Checks if vector store directory exists before loading
   - Allows dangerous deserialization with explicit flag

3. **Input Validation**:
   - Validates that at least one URL is provided
   - Validates that at least one document was successfully loaded
   - Validates question input before processing

---

## Interview Preparation Guide

### Common Questions & Answers

**Q: Why use FAISS instead of other vector databases?**
A: FAISS is lightweight, fast for similarity search, and works well for medium-scale applications. It's CPU-based and requires no external service, making it ideal for local development and small production deployments.

**Q: How does the application prevent AI hallucination?**
A: By using a "retrieval-augmented generation" approach where the LLM is instructed to answer ONLY based on provided context. The system prompt explicitly states "Answer using only the supplied context."

**Q: What's the advantage of chunk overlap?**
A: Ensures that information split across chunk boundaries isn't lost. A 200-character overlap guarantees that relevant context near boundaries is included in at least one chunk.

**Q: Why use a custom SimpleRetrievalQAWithSourcesChain instead of LangChain's built-in?**
A: Ensures compatibility with latest LangChain versions while customizing source tracking and response format for specific needs.

**Q: How does embedding work in this context?**
A: OpenAI's embeddings convert text into high-dimensional vectors representing semantic meaning. Similar text has similar vectors, enabling semantic search without keyword matching.

**Q: What's the purpose of the RecursiveCharacterTextSplitter?**
A: It splits text hierarchically (paragraphs → sentences → words) to preserve semantic coherence, unlike simple character splitting.

**Q: Why store vectors locally instead of using cloud services?**
A: Local FAISS storage reduces latency, eliminates API calls for retrieval, saves costs, and provides data privacy.

**Q: How does the system handle rate limiting or API failures?**
A: Currently, it doesn't implement retry logic or rate limiting. In production, you'd add exponential backoff and retry mechanisms.

**Q: What's the maximum context size the system can handle?**
A: Limited by OpenAI's token limits (8k for GPT-3.5, 128k for GPT-4). With 4 chunks of ~1000 chars, typically well within limits.

**Q: How would you scale this application?**
A: For scalability: use cloud vector DB (Pinecone/Weaviate), implement async processing, add caching, use connection pooling, and implement load balancing.

---

## Potential Improvements & Enhancements

1. **Advanced Features**:
   - Summarization of entire articles
   - Multi-turn conversation with memory
   - PDF/document upload support
   - Real-time news scraping

2. **Performance Optimizations**:
   - Implement async URL fetching
   - Add caching for embeddings
   - Use faster embedding models
   - Implement batch processing

3. **Reliability Improvements**:
   - Add retry logic with exponential backoff
   - Implement rate limiting
   - Add logging and monitoring
   - Health checks for API services

4. **User Experience**:
   - Save conversation history
   - Allow custom system prompts
   - Provide similarity scores with answers
   - Support multiple languages

5. **Production Readiness**:
   - Add authentication and authorization
   - Implement API rate limiting
   - Add audit logging
   - Use environment-specific configs
   - Add CI/CD pipeline

---

## Dependencies Overview

| Package | Version | Purpose |
|---------|---------|---------|
| langchain | 0.0.284 | LLM application framework |
| streamlit | 1.22.0 | Web UI framework |
| faiss-cpu | ≥1.7.4 | Vector similarity search |
| openai | 0.28.0 | API client for GPT and embeddings |
| unstructured | 0.9.2 | Document processing |
| tiktoken | 0.4.0 | Token counting |
| python-dotenv | 1.0.0 | Environment variable management |
| requests | (implicit) | HTTP requests for URL fetching |
| python-magic | 0.4.27 | File type detection |

---

## Setup & Deployment

### Local Development Setup
```bash
# Clone repository
git clone https://github.com/widushan/News-Research-Tool.git
cd News-Research-Tool

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "OPENAI_API_KEY=your_key_here" > .env

# Run application
streamlit run main.py
```

### Access Application
Navigate to `http://localhost:8501` in your browser

---

## File Structure

```
News-Research-Tool/
├── main.py                      # Main Streamlit application
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── .env                         # Environment variables (create this)
├── APPLICATION_DESCRIPTION.md   # This file
├── faiss_store_openai/          # Vector store directory (auto-generated)
│   └── index.faiss             # FAISS index file
├── notebooks/                   # Jupyter notebooks for experimentation
│   ├── faiss_tutorial.ipynb
│   ├── retrieval.ipynb
│   ├── text_loaders_splitters.ipynb
│   ├── movies.csv
│   ├── sample_text.csv
│   ├── nvda_news_1.txt
│   └── faiss_index/
│       └── index.faiss
└── Run Helper.txt              # Helper instructions

```

---

## Summary

The News Research Tool is a well-architected RAG (Retrieval-Augmented Generation) application that demonstrates:
- Effective integration of multiple AI technologies
- Clean separation of concerns
- Error handling and user experience considerations
- Efficient use of vector databases for semantic search
- Custom solutions compatible with evolving frameworks

This project serves as an excellent example of modern AI application development and is suitable for interview discussions about LLM applications, RAG systems, and practical AI deployment.
