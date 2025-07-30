# SimpleRAG-Ollama-ChromaDB

**SimpleRAG-Ollama-ChromaDB** is a simple implementation of a **Retrieval-Augmented Generation (RAG)** pipeline that combines **Ollama** for generating embeddings and a **ChromaDB** vector database to store and retrieve relevant documents based on a given query. This project demonstrates the power of combining vector databases and language models for intelligent document retrieval and generation.

## Features

- **Document Retrieval**: Retrieve relevant documents from ChromaDB based on a query.
- **Text Generation**: Use Ollama LLM (Language Model) to generate contextually relevant answers using the retrieved documents.
- **Easy Integration**: Integrates ChromaDB with Ollama for easy setup and use of RAG in your own projects.

## Requirements

- Python 3.x
- [Ollama](https://ollama.com) (make sure Ollama is running on your local machine)
- [ChromaDB](https://www.trychroma.com)
  
### Installing Dependencies

1. **Create a Virtual Environment**:
   ```bash
   python -m venv venv

2. **Activate the Virtual Environment**:
On Windows:
.\venv\Scripts\activate

3. **Install Required Libraries**:
This project includes a requirements.txt file to install all necessary dependencies. Simply run:

pip install -r requirements.txt

### Setup
### Run Ollama Model:

Ensure Ollama is running locally at the default endpoint (http://localhost:11434). If you are using a custom setup, update the base URL in the code.

### ChromaDB Setup:

ChromaDB will store the document embeddings locally. The default folder is chroma_db, but you can modify it as needed.
