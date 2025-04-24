# RAG Project

A Retrieval Augmented Generation (RAG) system using LangGraph, OpenAI and Gradio. from_rag_to_riches

## Features

- Document ingestion and chunking
- Vector embedding using OpenAI's models
- Document retrieval based on semantic similarity
- Text cleaning and rewriting
- LangGraph-based RAG implementation
- Gradio web interface
- Command-line interface

## Installation

```bash
# Clone the repository
git clone https://github.com/jcrigoni/from_rag_to_riches.git
cd from_rag_to_riches

# Install the package
pip install -e .
```

## Usage

### Command Line Interface

The project provides a command-line interface with the following commands:

#### Process Text Files

```bash
# Process text files
rag process --input-dir data/input_data --output-dir data/processed_data
```

#### Ingest Documents

```bash
# Ingest documents into the vector database
rag ingest --input-dir data/input_data
```

#### Launch Web Interface

```bash
# Launch the web interface
rag web
```

### Python API

```python
# Load and process documents
from rag_project.data.ingest import load_documents, create_chunks
from rag_project.core.embeddings import create_vectorstore

# Load documents
documents = load_documents("path/to/your/data")

# Create chunks
chunks = create_chunks(documents)

# Create vector database
vectorstore = create_vectorstore(chunks)

# Use the RAG chain
from rag_project.core.rag_graph import rag_chain
from langchain_core.messages import HumanMessage

# Query the RAG chain
response = rag_chain.invoke({
    "messages": [HumanMessage(content="Your question here")],
    "context": []
})

print(response["messages"][-1].content)
```

## Directory Structure

```
rag_project/
├── __init__.py
├── cli/
│   ├── __init__.py
│   └── main.py
├── config/
│   ├── __init__.py
│   └── settings.py
├── core/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── retriever.py
│   └── rag_graph.py
├── data/
│   ├── __init__.py
│   ├── ingest.py
│   └── processors.py
├── utils/
│   ├── __init__.py
│   └── text_utils.py
└── web/
    ├── __init__.py
    └── app.py
```

## Configuration

Edit `rag_project/config/settings.py` to configure:

- API keys
- Model settings
- Data directories
- Vector database settings

## License

[This project is licensed under the MIT License]