# Import required libraries
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import chromadb
import os

# Define the LLM model to be used
llm_model = "llama3.2"

# Initialize the ChromaDB client with persistent storage in the current directory
chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db"))

# Define a custom embedding function for ChromaDB using Ollama
class ChromaDBEmbeddingFunction:
    """
    Custom embedding function for ChromaDB using embeddings from Ollama.
    """
    def __init__(self, langchain_embeddings, name="ollama_embedding"):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        # Ensure the input is in a list format for processing
        if isinstance(input, str):
            input = [input]
        return self.langchain_embeddings.embed_documents(input)
    def name(self):
        # Return the name of the embedding function
        return self.name
# Initialize the embedding function with Ollama embeddings
embedding = ChromaDBEmbeddingFunction(
    OllamaEmbeddings(
        model=llm_model,
        base_url="http://localhost:11434"  # Adjust the base URL as per your Ollama server configuration
    )
)

# Define a collection for the RAG workflow
collection_name = "rag_collection_demo_1"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"description": "A collection for RAG with Ollama - Demo1"},
    embedding_function=embedding  # Use the custom embedding function
)

# Function to add documents to the ChromaDB collection
def add_documents_to_collection(documents, ids):
    """
    Add documents to the ChromaDB collection.
    
    Args:
        documents (list of str): The documents to add.
        ids (list of str): Unique IDs for the documents.
    """
    collection.add(
        documents=documents,
        ids=ids
    )

# Function to update and add documents to ChromaDB
def update_documents_in_chromadb(documents, doc_ids):
    """
    Update documents in ChromaDB. If documents exist, they will be replaced.
    
    Args:
        documents (list): The list of documents to add.
        doc_ids (list): The list of document IDs.
    """
    # Add documents to the ChromaDB collection
    collection.add(
        documents=documents,
        ids=doc_ids
    )
    print(f"Updated {len(documents)} documents in the collection.")

# Example of updated documents
documents = [
    "Artificial intelligence is the simulation of human intelligence processes by machines.",
    "Python is a programming language that lets you work quickly and integrate systems more effectively.",
    "Ala Ben Ayed is a software engineer with expertise in AI and machine learning, focusing on AI-based solutions."
]
doc_ids = ["doc1", "doc2", "doc4"]

# Update the documents in ChromaDB
update_documents_in_chromadb(documents, doc_ids)

# Function to query the ChromaDB collection
def query_chromadb(query_text, n_results=1):
    """
    Query the ChromaDB collection for relevant documents.
    
    Args:
        query_text (str): The input query.
        n_results (int): The number of top results to return.
    
    Returns:
        list of dict: The top matching documents and their metadata.
    """
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results["documents"], results["metadatas"]

# Function to interact with the Ollama LLM
def query_ollama(prompt):
    """
    Send a query to Ollama and retrieve the response.
    
    Args:
        prompt (str): The input prompt for Ollama.
    
    Returns:
        str: The response from Ollama.
    """
    llm = OllamaLLM(model=llm_model)
    return llm.invoke(prompt)

# RAG pipeline: Combine ChromaDB and Ollama for Retrieval-Augmented Generation
def rag_pipeline(query_text):
    """
    Perform Retrieval-Augmented Generation (RAG) by combining ChromaDB and Ollama.
    
    Args:
        query_text (str): The input query.
    
    Returns:
        str: The generated response from Ollama augmented with retrieved context.
    """
    # Step 1: Retrieve relevant documents from ChromaDB
    retrieved_docs, metadata = query_chromadb(query_text)
    context = " ".join(retrieved_docs[0]) if retrieved_docs else "No relevant documents found."

    # Step 2: Send the query along with the context to Ollama
    augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
    print("######## Augmented Prompt ########")
    print(augmented_prompt)

    response = query_ollama(augmented_prompt)
    return response
# Function to view all documents in the ChromaDB collection without querying

def view_all_documents_in_chromadb():
    """
    Retrieve and display all documents from the ChromaDB collection without querying.
    """
    # Get the documents in the collection
    documents = collection.get()  # Include metadata if needed
    
    # Print the documents and their metadata
    print("######## Retrieved Documents ########")
    for doc, metadata in zip(documents["documents"], documents["metadatas"]):
        print(f"Document: {doc}")
        print(f"Metadata: {metadata}")

# Example usage: View all documents in the collection
view_all_documents_in_chromadb()

# Example usage
# Define a query to test the RAG pipeline
query = "Who is ala ben ayed ?"  # Change the query as needed
response = rag_pipeline(query)
print("######## Response from LLM ########\n", response)