"""Configuration settings for the RAG project."""
import os

# OpenAI API key
OPENAI_API_KEY = os.getenv("MY_OPENAI_API_KEY")

# Vector database settings
VECTORSTORE_DIR = "vectorstore"

# Embedding model settings (OpenAI)
EMBEDDING_MODEL = "text-embedding-3-small"

# Rewriter LLM settings
LLM_MODEL = "gpt-4.1-nano"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 3000

# Retriever settings
DEFAULT_TOP_K = 5

# Data directories
INPUT_DATA_DIR = "data/input_data"  # Raw markdown files
PROCESSED_DATA_DIR = "data/processed_data" # Processed markdown files
TEST_DATA_DIR = "data/test_data"