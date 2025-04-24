"""Command-line interface for the RAG application. from_rag_to_riches"""

import argparse
import sys
import textwrap
import os
from rag_project.data.ingest import load_documents, create_chunks
from rag_project.core.embeddings import create_vectorstore
from rag_project.data.processors import process_files
from rag_project.config import settings

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RAG Project CLI",
        epilog="For detailed examples, use --help-examples or --examples"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into the vector database")
    ingest_parser.add_argument("--input-dir", help="Directory containing input files", default=settings.INPUT_DATA_DIR)
    ingest_parser.add_argument("--chunk-size", type=int, help="Size of document chunks", default=500)
    ingest_parser.add_argument("--chunk-overlap", type=int, help="Overlap between chunks", default=100)
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process text files")
    process_parser.add_argument("--input-dir", help="Directory containing input files", default=settings.INPUT_DATA_DIR)
    process_parser.add_argument("--output-dir", help="Directory to save processed files", default=settings.PROCESSED_DATA_DIR)
    process_parser.add_argument("--clean-only", action="store_true", help="Only clean text without rewriting")
    
    # Web interface command
    web_parser = subparsers.add_parser("web", help="Launch web interface")
    
    return parser.parse_args()

def ingest_command(args):
    """
    Ingest documents and create vector database.
    
    Args:
        args: Command line arguments
    """
    print(f"Loading documents from {args.input_dir}...")
    documents = load_documents(directory=args.input_dir)
    
    print(f"Creating chunks with size={args.chunk_size}, overlap={args.chunk_overlap}...")
    chunks = create_chunks(
        documents=documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    print("Creating vector database...")
    create_vectorstore(documents=chunks)
    
    print("Ingestion complete!")

def process_command(args):
    """
    Process text files.
    
    Args:
        args: Command line arguments
    """
    print(f"Processing files from {args.input_dir} to {args.output_dir}...")
    num_files = process_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        clean_only=args.clean_only
    )
    print(f"Processed {num_files} files!")

def web_command(args):
    """
    Launch web interface.
    
    Args:
        args: Command line arguments
    """
    from rag_project.web.app import launch_app
    
    print("Launching web interface...")
    launch_app()

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

def main():
    """Main entry point for the CLI."""
    # Check if the user requests custom help
    if len(sys.argv) > 1 and sys.argv[1] in ['--help-examples', '--examples']:
        print_custom_help()
        return
        
    args = parse_args()
    
    # Execute the appropriate command
    if args.command == "ingest":
        ingest_command(args)
    elif args.command == "process":
        process_command(args)
    elif args.command == "web":
        web_command(args)
    else:
        print("Please specify a command. Use --help for basic options or --help-examples for detailed examples.")
        sys.exit(1)

if __name__ == "__main__":
    main()