# AmbedkarGPT - RAG-Based Question Answering System

## Overview

**AmbedkarGPT** is a Retrieval-Augmented Generation (RAG) system that leverages local LLMs and embedding models to answer questions based on the content of text documents. The system implements a full RAG pipeline using LangChain, ChromaDB, and Ollama, demonstrating how to build intelligent question-answering applications without relying on cloud-based APIs.

## Project Architecture

### Components

1. **Document Loading** (`TextLoader`)
   - Loads and reads text files for processing
   - Handles file I/O with error handling

2. **Text Splitting** (`CharacterTextSplitter`)
   - Splits documents into manageable chunks (500 characters with 50-character overlap)
   - Maintains context across chunks through overlapping

3. **Embeddings** (`HuggingFaceEmbeddings`)
   - Uses `sentence-transformers/all-MiniLM-L6-v2` model
   - Converts text chunks into dense vector representations
   - Runs locally without external API calls

4. **Vector Database** (`ChromaDB`)
   - Stores document embeddings locally
   - Enables fast similarity search and retrieval
   - Persistent local storage of vector representations

5. **Language Model** (`Ollama`)
   - Runs Mistral 7B model locally
   - Generates context-aware responses
   - Eliminates dependency on cloud LLM services

6. **Retrieval Pipeline**
   - Custom retrieval function that:
     - Retrieves top-3 most relevant documents based on query similarity
     - Constructs a contextual prompt with retrieved documents
     - Generates answers using the LLM with provided context

## Features

- ✅ **Local Execution**: Runs entirely on-machine without external API dependencies
- ✅ **RAG Pipeline**: Combines retrieval with generation for factually grounded responses
- ✅ **Interactive Interface**: Command-line conversation interface
- ✅ **Error Handling**: Robust exception handling for file operations and inference
- ✅ **Context-Aware**: Retrieves relevant documents before generation
- ✅ **Easy Exit**: Type 'exit', 'quit', or 'q' to terminate the program

## Installation

### Prerequisites
- Python 3.8 or higher
- Ollama installed and running with Mistral 7B model
- Sufficient disk space for embeddings model (~200MB)

### Setup Steps

1. **Clone/Download the project**
   ```bash
   cd "intern assingment"
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure Ollama is running**
   ```bash
   ollama serve
   ```
   In another terminal, pull the Mistral model:
   ```bash
   ollama pull mistral
   ```

## Usage

Run the application:
```bash
python main.py
```

### Sample Interaction

```
Loading data...
Splitting text...
Creating embeddings and vector store...
Initializing Ollama (Mistral)...
Building RAG pipeline...

--- AmbedkarGPT Initialized ---
Ask a question based on the speech (Type 'exit' to quit):

Your Question: What is the main problem discussed?
Answer: The main problem discussed is the issue of caste and the role of shastras (religious scriptures) in perpetuating it. According to the text, the real enemy is the belief in the sanctity of the shastras, as long as people believe in them, they will never be able to get rid of caste.

Your Question: exit
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` | Core framework for building LLM applications |
| `langchain-community` | Community integrations (Ollama, Chroma) |
| `langchain-huggingface` | HuggingFace embeddings integration |
| `chromadb` | Vector database for storing embeddings |
| `sentence-transformers` | Pre-trained embedding models |
| `ollama` | Local LLM integration |

## How It Works

### Step-by-Step Process

1. **Loading Phase**
   - Reads `speech.txt` file containing source material
   - Handles file not found errors gracefully

2. **Preprocessing Phase**
   - Splits text into 500-character chunks with 50-character overlap
   - Overlap helps maintain context between chunks

3. **Embedding Phase**
   - Converts each chunk into embeddings using `sentence-transformers/all-MiniLM-L6-v2`
   - Stores embeddings in ChromaDB for fast retrieval

4. **Retrieval Phase**
   - When user asks a question, the system:
     - Converts query to embedding using same model
     - Searches for top-3 most similar document chunks
     - Retrieves relevant context

5. **Generation Phase**
   - Constructs a prompt with retrieved context and user query
   - Sends to Mistral 7B LLM via Ollama
   - Returns generated answer

## Configuration

### Customizable Parameters

In `main.py`, you can adjust:

- **Chunk Size**: Line 26
  ```python
  text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
  ```

- **Number of Retrieved Documents**: Line 40
  ```python
  retriever = db.as_retriever(search_kwargs={"k": 3})
  ```

- **Embedding Model**: Line 33
  ```python
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  ```

- **LLM Model**: Line 36
  ```python
  llm = Ollama(model="mistral:latest")
  ```

## Document: speech.txt

The application currently processes a speech that discusses:
- The problem of caste in society
- The role of religious scriptures (shastras) in perpetuating caste
- The need to challenge the sanctity of scriptures as a path to social reform

This can be replaced with any text file relevant to your use case.

## Error Handling

The application handles:
- ❌ Missing `speech.txt` file
- ❌ Embedding model download failures
- ❌ Ollama connection issues
- ❌ Invalid or problematic queries
- ❌ General inference errors

All errors are caught and displayed to the user with descriptive messages.

## Performance Considerations

- **First Run**: ~1-2 minutes (downloads embedding model)
- **Initialization**: ~10-30 seconds (loads model and builds vector DB)
- **Per Query**: ~2-5 seconds (retrieval + generation)

### Optimization Tips
- Increase `chunk_size` for fewer, larger context windows
- Decrease `k` in retriever to use fewer documents (faster but less context)
- Use a smaller LLM model (e.g., Ollama's default models)

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'langchain_community'"
**Solution**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

### Issue: "Ollama connection refused"
**Solution**: Start Ollama service
```bash
ollama serve
```

### Issue: "Model mistral not found"
**Solution**: Pull the model
```bash
ollama pull mistral
```

### Issue: Slow response times
**Solution**: 
- Check if Ollama is running on a CPU (GPU significantly faster)
- Use fewer retrieved documents (`k=1` or `k=2`)
- Increase chunk size to reduce number of chunks

## Future Enhancements

- [ ] Support for multiple documents/knowledge bases
- [ ] Persistent vector store (avoid re-embedding on startup)
- [ ] Web interface (Flask/Streamlit)
- [ ] Multi-turn conversation with memory
- [ ] Document upload functionality
- [ ] Response citation tracking
- [ ] Configurable LLM and embedding model selection

## References

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Ollama Models](https://ollama.ai/)
- [Sentence Transformers](https://www.sbert.net/)

## License

This project is provided for educational purposes.

## Notes

- The application uses `sentence-transformers/all-MiniLM-L6-v2`, a lightweight yet effective embedding model (~28MB)
- Mistral 7B is a 7-billion parameter model optimized for local deployment
- All processing happens locally with no data sent to external servers
- The vector database is created in memory and will be lost after program termination (can be persisted with Chroma persistence settings)

---

**Last Updated**: December 7, 2025  
**Author**: Intern Assignment  
**Status**: ✅ Fully Functional
