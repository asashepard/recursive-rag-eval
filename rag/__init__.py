"""
rag - Regulatory RAG Pipeline
------------------------------
Structured retrieval-augmented generation for utility regulatory compliance.
"""

from .schemas import (
    Citation,
    Obligation,
    StateObligations,
    DiffItem,
    ActivityResponse,
    ACTIVITY_RESPONSE_SCHEMA,
)
from .chunker import Chunk, build_chunks, load_chunks
from .embeddings import embed_query, load_embeddings
from .index import VectorIndex, load_index
from .bm25_search import BM25Index, load_bm25_index
from .retriever import Retriever

__all__ = [
    # Schemas
    "Citation",
    "Obligation", 
    "StateObligations",
    "DiffItem",
    "ActivityResponse",
    "ACTIVITY_RESPONSE_SCHEMA",
    # Chunking
    "Chunk",
    "build_chunks",
    "load_chunks",
    # Embeddings
    "embed_query",
    "load_embeddings",
    # Index
    "VectorIndex",
    "load_index",
    # BM25
    "BM25Index",
    "load_bm25_index",
    # Retrieval
    "Retriever",
]
