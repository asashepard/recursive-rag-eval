"""
bm25_search.py
--------------
BM25 (keyword-based) retrieval using rank_bm25.
Hybrid with semantic search to improve recall on regulatory language.
"""

import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from .chunker import Chunk, load_chunks

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CORPUS_ROOT = Path(__file__).resolve().parent.parent / "corpus" / "normalized"
BM25_INDEX_PATH = CORPUS_ROOT / "bm25_index.json"


class BM25Index:
    """
    BM25 index for keyword-based retrieval.
    """
    
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: list[Chunk] = []
        self.tokenized_texts: list[list[str]] = []
    
    def build(self, chunks: list[Chunk]):
        """Build BM25 index from chunks."""
        self.chunks = chunks
        
        # Tokenize chunks (simple whitespace tokenization)
        self.tokenized_texts = [
            chunk.text.lower().split()
            for chunk in chunks
        ]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_texts)
        print(f"Built BM25 index with {len(chunks)} documents")
    
    def search(
        self,
        query: str,
        top_k: int = 8,
        states: list[str] | None = None,
        utility_domain: str | None = "electric",
        include_federal: bool = True,
    ) -> list[tuple[Chunk, float]]:
        """
        BM25 search with optional metadata filtering.
        """
        query_tokens = query.lower().split()
        
        # Score all documents
        scores = self.bm25.get_scores(query_tokens)
        
        # Apply metadata filter
        mask = [True] * len(self.chunks)
        
        if states is not None:
            allowed_states = set(states)
            if include_federal:
                allowed_states.add("FED")
                allowed_states.add("NARUC")
            
            for i, chunk in enumerate(self.chunks):
                if chunk.state not in allowed_states:
                    mask[i] = False
        
        if utility_domain:
            for i, chunk in enumerate(self.chunks):
                if chunk.utility_domain != utility_domain and chunk.utility_domain != "general":
                    mask[i] = False
        
        # Collect results
        results = []
        for i, (chunk, score) in enumerate(zip(self.chunks, scores)):
            if mask[i] and score > 0:
                results.append((chunk, float(score)))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def save(self, path: Path = BM25_INDEX_PATH):
        """Save index to disk (chunks only; BM25 is small)."""
        data = {
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "state": c.state,
                    "text": c.text[:500],  # Store first 500 chars for debugging
                }
                for c in self.chunks
            ]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"Saved BM25 index metadata to {path}")
    
    def load(self, chunks: list[Chunk], path: Path = BM25_INDEX_PATH):
        """Load from chunks (rebuild is fast)."""
        self.build(chunks)


def build_bm25_index():
    """Build and save BM25 index."""
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")
    
    index = BM25Index()
    index.build(chunks)
    index.save()
    
    return index


def load_bm25_index(chunks: list[Chunk]) -> BM25Index:
    """Load BM25 index."""
    index = BM25Index()
    index.build(chunks)
    return index


if __name__ == "__main__":
    build_bm25_index()
