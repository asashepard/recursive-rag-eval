"""
embeddings.py
-------------
Embed chunks using a local sentence-transformer model.
Using BGE-large-en for good quality without API costs.
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .chunker import Chunk, load_chunks, CHUNKS_OUT

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# BGE-large-en: strong open model, 1024-dim embeddings
MODEL_NAME = "BAAI/bge-large-en-v1.5"
CORPUS_ROOT = Path(__file__).resolve().parent.parent / "corpus" / "normalized"
EMBEDDINGS_OUT = CORPUS_ROOT / "embeddings.npy"
CHUNK_IDS_OUT = CORPUS_ROOT / "chunk_ids.json"

# Batch size for encoding
BATCH_SIZE = 32


def load_model() -> SentenceTransformer:
    """Load the embedding model."""
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    return model


def embed_chunks(chunks: list[Chunk], model: SentenceTransformer) -> np.ndarray:
    """
    Embed all chunks and return matrix of shape (n_chunks, embed_dim).
    BGE models work best with instruction prefix for retrieval.
    """
    # For BGE, prepend instruction for better retrieval
    texts = [
        f"Represent this regulatory document for retrieval: {c.text}"
        for c in chunks
    ]
    
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2 normalize for cosine sim via dot product
    )
    
    return np.array(embeddings, dtype=np.float32)


def save_embeddings(embeddings: np.ndarray, chunk_ids: list[str]):
    """Save embeddings matrix and chunk ID mapping."""
    np.save(EMBEDDINGS_OUT, embeddings)
    with open(CHUNK_IDS_OUT, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f)
    print(f"Saved embeddings: {EMBEDDINGS_OUT}")
    print(f"Saved chunk IDs: {CHUNK_IDS_OUT}")


def load_embeddings() -> tuple[np.ndarray, list[str]]:
    """Load embeddings and chunk ID mapping."""
    embeddings = np.load(EMBEDDINGS_OUT)
    with open(CHUNK_IDS_OUT, "r", encoding="utf-8") as f:
        chunk_ids = json.load(f)
    return embeddings, chunk_ids


def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    """
    Embed a query. BGE uses different prefix for queries.
    """
    # BGE query instruction
    text = f"Represent this question for retrieving relevant regulatory documents: {query}"
    embedding = model.encode(
        [text],
        normalize_embeddings=True,
    )
    return np.array(embedding, dtype=np.float32)


if __name__ == "__main__":
    # Load chunks
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")
    
    # Load model and embed
    model = load_model()
    embeddings = embed_chunks(chunks, model)
    
    # Save
    chunk_ids = [c.chunk_id for c in chunks]
    save_embeddings(embeddings, chunk_ids)
    
    print(f"\nEmbedding matrix shape: {embeddings.shape}")
