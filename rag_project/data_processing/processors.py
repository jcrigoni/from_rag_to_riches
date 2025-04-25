"""Module for processing and transforming document data."""

import os
from rag_project.utils.text_utils import clean_text, rewrite_text
from rag_project.config import settings

def process_files(input_dir=None, output_dir=None, clean_only=False):
    """
    Process markdown files by cleaning and optionally rewriting them.
    
    Args:
        input_dir (str): Directory containing input files
        output_dir (str): Directory to save processed files
        clean_only (bool): If True, only clean text without rewriting
        
    Returns:
        int: Number of files processed
    """
    if input_dir is None:
        input_dir = settings.INPUT_DATA_DIR
        
    if output_dir is None:
        output_dir = settings.PROCESSED_DATA_DIR
    
    # Ensure directories exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    files_processed = 0
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".md"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Read the file
            with open(input_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Clean the text
            cleaned_text = clean_text(content)
            
            # Optionally rewrite the text
            if not clean_only:
                processed_text = rewrite_text(cleaned_text)
            else:
                processed_text = cleaned_text
            
            # Write the processed text
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(processed_text)
            
            files_processed += 1
            print(f"✅ Processed file: {filename}")
    
    return files_processed