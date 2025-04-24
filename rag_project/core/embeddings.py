"""Module for handling document embeddings and vector storage."""

import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from rag_project.config import settings

def get_embedding_model(model_name=None):
    """
    Get an initialized embedding model.
    
    Args:
        model_name (str): Name of the embedding model to use
        
    Returns:
        OpenAIEmbeddings: Initialized embedding model
    """
    if model_name is None:
        model_name = settings.EMBEDDING_MODEL
    
    return OpenAIEmbeddings(model=model_name)

def create_vectorstore(documents, persist_directory=None, embedding_model=None):
    """
    Create a vector database from documents.
    
    Args:
        documents (list): List of documents to embed
        persist_directory (str): Directory to save vector database
        embedding_model: Embedding model to use
        
    Returns:
        Chroma: Vector database
    """
    if persist_directory is None:
        persist_directory = settings.VECTORSTORE_DIR
    
    if embedding_model is None:
        embedding_model = get_embedding_model()
    
    # Create directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)
    
    # Create and persist vector database
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    
    # Persist the database
    vectorstore.persist()
    
    print(f"Vector database created and saved in {persist_directory}")
    return vectorstore

def load_vectorstore(persist_directory=None, embedding_model=None):
    """
    Load a vector database from disk.
    
    Args:
        persist_directory (str): Directory containing vector database
        embedding_model: Embedding model to use
        
    Returns:
        Chroma: Vector database
    """
    if persist_directory is None:
        persist_directory = settings.VECTORSTORE_DIR
    
    if embedding_model is None:
        embedding_model = get_embedding_model()
    
    # Load the vector database
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
    
    return vectorstore