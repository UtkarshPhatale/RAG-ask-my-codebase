"""
Knowledge Base

Handles document loading, chunking, embeddings, and vector store
creation for the RAG pipeline.
"""

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings():
    """Return a local HuggingFace embedding model (runs on CPU, no API key)."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def load_and_chunk(data_dir: str, chunk_size=500, chunk_overlap=50):
    """Load .txt files from data_dir and split into chunks."""
    loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    return chunks


def build_knowledge_base(data_dir: str):
    """Build a FAISS vector store from documents in data_dir.

    Returns:
        A FAISS vector store you can search with .similarity_search(query, k=3)
    """
    print("Loading documents...")
    chunks = load_and_chunk(data_dir)
    print(f"  Split into {len(chunks)} chunks")

    print("Building vector store (first run downloads ~80MB model)...")
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("  Done!\n")

    return vector_store
