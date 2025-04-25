"""Module for loading and processing documents."""

import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_project.config import settings

def load_documents(directory=None, glob_pattern="**/*.md", show_progress=True):
    """
    Load documents from a directory.
    
    Args:
        directory (str): Directory to load documents from. If None, uses default from settings.
        glob_pattern (str): Pattern to match files
        show_progress (bool): Whether to show progress during loading
        
    Returns:
        list: List of loaded documents
    """
    if directory is None:
        directory = settings.INPUT_DATA_DIR
        
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Initialize document loader
    loader = DirectoryLoader(directory, glob=glob_pattern, show_progress=show_progress)
    
    # Load documents
    documents = loader.load()
    
    print(f"Loading of {len(documents)} documents done")
    return documents

def create_chunks(documents, chunk_size=500, chunk_overlap=100):
    """
    Split documents into smaller chunks.
    
    Args:
        documents (list): List of documents to split
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        list: List of document chunks
    """
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    # Split documents into chunks
    chunks = text_splitter.split_documents(documents)
    
    print(f"Creation of {len(chunks)} chunks done")
    return chunks