"""Module for LangGraph-based RAG implementation. """

from typing import TypedDict, Annotated, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from rag_project.core.retriever import DocumentRetriever
from rag_project.config import settings

# Define the state type for the RAG chain
class RAGState(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage]], add_messages]
    context: List[Document]

# Initialize the document retriever
doc_retriever = DocumentRetriever(
    persist_directory=settings.VECTORSTORE_DIR, 
    top_k=10
)

def retrieve(state: RAGState) -> RAGState:
    """
    Retrieve relevant documents based on the user's message.
    
    Args:
        state (RAGState): Current state of the conversation
    
    Returns:
        RAGState: Updated state with retrieved documents
    """
    # Get the last message from the state
    last_message = state["messages"][-1]
    if not isinstance(last_message, HumanMessage):
        return {"context": []}
    
    # Retrieve relevant documents using the retriever
    docs = doc_retriever.get_relevant_documents(last_message.content)
    return {"context": docs}

def generate(state: RAGState) -> RAGState:
    """
    Generate a response based on the retrieved documents.
    
    Args:
        state (RAGState): Current state with context and messages
    
    Returns:
        RAGState: Updated state with AI response
    """
    # Initialize the model
    llm = ChatOpenAI(
        temperature=settings.LLM_TEMPERATURE, 
        model=settings.LLM_MODEL, 
        max_tokens=settings.LLM_MAX_TOKENS
    )
    
    # Prepare context for the LLM
    context_str = "\n\n".join([doc.page_content for doc in state["context"]])
    
    # Prepare prompt with context
    last_message = state["messages"][-1]
    augmented_prompt = f"""
    Tu es un assistant spécialisé UNIQUEMENT sur les documents qui te sont fournis. 
    Tu dois STRICTEMENT te limiter à ces informations.

    Contexte fourni:
    {context_str}

    Question de l'utilisateur: {last_message.content}

    INSTRUCTIONS IMPORTANTES:
    1. ANALYSE SI la question peut être répondue avec les informations du contexte ci-dessus.
    2. Si tu peux répondre avec ces informations, fais-le en te basant EXCLUSIVEMENT sur le contexte.
    3. Si tu ne peux PAS répondre avec ces informations, réponds UNIQUEMENT: "Je ne dispose pas d'informations sur ce sujet dans ma base de connaissances actuelle."
    
    Réponse:
    """
    
    # Generate response
    response = llm.invoke(augmented_prompt)
    
    return {"messages": [AIMessage(content=response.content)]}

# Create RAG chain
def create_rag_chain():
    """
    Create and compile the RAG chain.
    
    Returns:
        StateGraph: Compiled RAG chain
    """
    # Define the graph
    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    
    # Define transitions
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    
    # Compile the graph
    return builder.compile()

# Create the RAG chain
rag_chain = create_rag_chain()