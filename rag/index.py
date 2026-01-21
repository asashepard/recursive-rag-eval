"""
index.py
--------
FAISS vector index with metadata filtering support.
"""

import json
from pathlib import Path
from dataclasses import asdict

import faiss
import numpy as np

from .chunker import Chunk, load_chunks
from .embeddings import load_embeddings

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CORPUS_ROOT = Path(__file__).resolve().parent.parent / "corpus" / "normalized"
INDEX_PATH = CORPUS_ROOT / "faiss.index"
CHUNK_LOOKUP_PATH = CORPUS_ROOT / "chunk_lookup.json"


class VectorIndex:
    """
    FAISS index wrapper with metadata filtering.
    Uses IndexFlatIP (inner product = cosine similarity for normalized vectors).
    """
    
    def __init__(self):
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[Chunk] = []
        self.chunk_id_to_idx: dict[str, int] = {}
    
    def build(self, chunks: list[Chunk], embeddings: np.ndarray):
        """Build index from chunks and their embeddings."""
        assert len(chunks) == embeddings.shape[0], "Mismatch between chunks and embeddings"
        
        self.chunks = chunks
        self.chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(chunks)}
        
        # Create FAISS index (inner product for normalized vectors = cosine)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        
        print(f"Built FAISS index with {self.index.ntotal} vectors, dim={dim}")
    
    def save(self):
        """Persist index and chunk lookup to disk."""
        faiss.write_index(self.index, str(INDEX_PATH))
        
        # Save chunk data for lookup
        lookup = {c.chunk_id: asdict(c) for c in self.chunks}
        with open(CHUNK_LOOKUP_PATH, "w", encoding="utf-8") as f:
            json.dump(lookup, f)
        
        print(f"Saved index to {INDEX_PATH}")
        print(f"Saved chunk lookup to {CHUNK_LOOKUP_PATH}")
    
    def load(self):
        """Load index and chunk lookup from disk."""
        self.index = faiss.read_index(str(INDEX_PATH))
        
        with open(CHUNK_LOOKUP_PATH, "r", encoding="utf-8") as f:
            lookup = json.load(f)
        
        self.chunks = [Chunk(**data) for data in lookup.values()]
        self.chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        
        print(f"Loaded FAISS index with {self.index.ntotal} vectors")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 8,
        states: list[str] | None = None,
        utility_domain: str | None = "electric",
        include_federal: bool = True,
    ) -> list[tuple[Chunk, float]]:
        """
        Search with optional metadata filtering.
        
        Args:
            query_embedding: (1, dim) or (dim,) query vector
            top_k: number of results to return
            states: filter to these states (e.g., ["TX", "OK"])
            utility_domain: filter by domain (default: "electric")
            include_federal: also include federal/NERC documents
        
        Returns:
            List of (Chunk, score) tuples, sorted by score descending
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Build filter mask
        mask = np.ones(len(self.chunks), dtype=bool)
        
        if states is not None:
            allowed_states = set(states)
            if include_federal:
                allowed_states.add("FED")
                allowed_states.add("NARUC")  # Reference material
            
            for i, chunk in enumerate(self.chunks):
                if chunk.state not in allowed_states:
                    mask[i] = False
        
        if utility_domain:
            for i, chunk in enumerate(self.chunks):
                if chunk.utility_domain != utility_domain and chunk.utility_domain != "general":
                    mask[i] = False
        
        # Get indices that pass filter
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) == 0:
            return []
        
        # For filtered search, we need to search more broadly then filter
        # (FAISS doesn't support native filtering on IndexFlatIP)
        search_k = min(top_k * 10, self.index.ntotal)  # Over-fetch
        
        scores, indices = self.index.search(query_embedding, search_k)
        scores = scores[0]
        indices = indices[0]
        
        # Filter and collect results
        results = []
        valid_set = set(valid_indices)
        
        for score, idx in zip(scores, indices):
            if idx in valid_set:
                results.append((self.chunks[idx], float(score)))
                if len(results) >= top_k:
                    break
        
        return results


def build_index():
    """Build and save the FAISS index."""
    # Load chunks
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")
    
    # Load embeddings
    embeddings, chunk_ids = load_embeddings()
    print(f"Loaded embeddings: {embeddings.shape}")
    
    # Verify alignment
    assert len(chunks) == len(chunk_ids), "Chunk count mismatch"
    for i, (chunk, cid) in enumerate(zip(chunks, chunk_ids)):
        assert chunk.chunk_id == cid, f"Chunk ID mismatch at {i}"
    
    # Build index
    index = VectorIndex()
    index.build(chunks, embeddings)
    index.save()
    
    return index


def load_index() -> VectorIndex:
    """Load the FAISS index from disk."""
    index = VectorIndex()
    index.load()
    return index


if __name__ == "__main__":
    build_index()
