"""Module for retrieving relevant documents."""

from typing import List
from langchain.schema import Document
from rag_project.core.embeddings import load_vectorstore, get_embedding_model
from rag_project.config import settings

class DocumentRetriever:
    """Class for retrieving relevant documents from a vector database."""
    
    def __init__(self, persist_directory=None, top_k=None):
        """
        Initialize the document retriever.
        
        Args:
            persist_directory (str): Directory of the Chroma vector database
            top_k (int): Number of documents to retrieve
        """
        if persist_directory is None:
            persist_directory = settings.VECTORSTORE_DIR
            
        if top_k is None:
            top_k = settings.DEFAULT_TOP_K
            
        self.top_k = top_k
        
        # Initialize embedding model
        self.embedding_model = get_embedding_model()
        
        # Load vector database
        self.vectorstore = load_vectorstore(
            persist_directory=persist_directory,
            embedding_model=self.embedding_model
        )
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.top_k}
        )
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get relevant documents for a query.
        
        Args:
            query (str): User query
            
        Returns:
            List[Document]: List of relevant documents
        """
        return self.retriever.get_relevant_documents(query)
    
    def similarity_search_with_score(self, query: str, k=None) -> List[tuple]:
        """
        Perform similarity search with scores.
        
        Args:
            query (str): User query
            k (int, optional): Number of results to return
            
        Returns:
            List[tuple]: List of tuples (document, score)
        """
        if k is None:
            k = self.top_k
            
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def update_retrieval_parameters(self, top_k=None, search_type=None, **kwargs):
        """
        Update retriever parameters.
        
        Args:
            top_k (int, optional): New number of documents to retrieve
            search_type (str, optional): Type of search ('similarity', 'mmr', etc.)
            **kwargs: Additional parameters
        """
        search_kwargs = {"k": top_k if top_k else self.top_k}
        search_kwargs.update(kwargs)
        
        if search_type:
            self.retriever = self.vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            )
        else:
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs=search_kwargs
            )
        
        if top_k:
            self.top_k = top_k