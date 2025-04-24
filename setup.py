"""Setup script for the rag_project package.from_rag_to_riches"""

from setuptools import setup, find_packages

setup(
    name="rag_project",
    version="0.1.0",
    description="RAG (Retrieval Augmented Generation) system using LangGraph",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-core",
        "langchain-community",
        "langchain-openai",
        "chromadb",
        "openai",
        "gradio",
        "langgraph",
        "unstructured[md]",
    ],
    entry_points={
        "console_scripts": [
            "rag=rag_project.cli.main:main",
        ],
    },
)