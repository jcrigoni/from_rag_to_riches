#!/usr/bin/env python3
"""Test script for the custom help function.from_rag_to_riches"""

import os
import sys
import textwrap

# Mock settings for testing
class Settings:
    INPUT_DATA_DIR = "data/input_data"
    PROCESSED_DATA_DIR = "data/processed_data"

settings = Settings()

def print_custom_help():
    """
    Print custom help message with usage examples.
    
    This function provides detailed examples of how to use each command
    in the RAG project, making it easier for users to get started.
    """
    help_text = """
    ╭────────────────────────────────────────────────────────────────────────╮
    │                        RAG PROJECT USER GUIDE                          │
    ╰────────────────────────────────────────────────────────────────────────╯
    
    This RAG (Retrieval Augmented Generation) tool offers the following commands:
    
    ╭─────────────────────╮
    │  1. PROCESS COMMAND │
    ╰─────────────────────╯
    
    Process and clean text files, optionally rewriting them with OpenAI.
    
    Example:
      rag process --input-dir data/input_data --output-dir data/processed_data
      
    Options:
      --input-dir   : Directory with markdown files (default: {})
      --output-dir  : Directory to save processed files (default: {})
      --clean-only  : Only clean text without rewriting
    
    ╭────────────────────╮
    │  2. INGEST COMMAND │
    ╰────────────────────╯
    
    Process documents and create a vector database for RAG.
    
    Example:
      rag ingest --input-dir data/input_data --chunk-size 500 --chunk-overlap 100
      
    Options:
      --input-dir     : Directory with markdown files (default: {})
      --chunk-size    : Size of text chunks (default: 500)
      --chunk-overlap : Overlap between chunks (default: 100)
    
    ╭─────────────────╮
    │  3. WEB COMMAND │
    ╰─────────────────╯
    
    Launch the Gradio web interface to interact with the RAG system.
    
    Example:
      rag web
    """.format(
        settings.INPUT_DATA_DIR,
        settings.PROCESSED_DATA_DIR,
        settings.INPUT_DATA_DIR
    )
    
    # Wrap the text to fit the terminal width
    try:
        terminal_width = os.get_terminal_size().columns
    except (OSError, AttributeError):
        # Default to 80 columns if terminal size can't be determined
        terminal_width = 80
        
    wrapped_text = ""
    for line in help_text.split('\n'):
        if line.strip().startswith('│') or line.strip().startswith('╭') or line.strip().startswith('╰'):
            # Don't wrap box drawing characters
            wrapped_text += line + '\n'
        else:
            wrapped_text += textwrap.fill(line, width=min(terminal_width, 80), 
                                        subsequent_indent='  ' if line.startswith('  ') else '') + '\n'
    
    print(wrapped_text)

if __name__ == "__main__":
    print_custom_help()