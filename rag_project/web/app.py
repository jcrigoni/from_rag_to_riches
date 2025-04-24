"""Gradio web interface for the RAG application."""

import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from rag_project.core.rag_graph import rag_chain

def chat(message, history):
    """
    Chat function for Gradio interface.
    
    Args:
        message (str): User message
        history (list): Chat history
        
    Returns:
        str: AI response
    """
    # Format history for RAG
    formatted_history = []
    for human, ai in history:
        formatted_history.append(HumanMessage(content=human))
        formatted_history.append(AIMessage(content=ai))
    
    # Add the new message
    current_message = HumanMessage(content=message)
    formatted_history.append(current_message)
    
    # Invoke the RAG chain
    response = rag_chain.invoke({"messages": formatted_history, "context": []})
    
    # Get the response
    ai_response = response["messages"][-1].content
    
    return ai_response

def create_demo():
    """
    Create and return the Gradio interface.
    
    Returns:
        gr.ChatInterface: Gradio chat interface
    """
    demo = gr.ChatInterface(
        chat,
        title="Chatbot RAG based on documents",
        description="Ask questions based on markdown documents"
    )
    return demo

def launch_app():
    """Launch the Gradio app."""
    demo = create_demo()
    demo.launch(share=True)

if __name__ == "__main__":
    launch_app()