"""
retriever.py
------------
Hybrid retrieval: semantic (embeddings) + lexical (BM25).
Increased recall, explicit federal overlays, optional reranking.
"""

from sentence_transformers import SentenceTransformer, CrossEncoder

from .chunker import Chunk
from .embeddings import embed_query, MODEL_NAME
from .index import load_index
from .bm25_search import load_bm25_index

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Cross-encoder for reranking (optional but improves precision)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Retriever:
    """
    Hybrid RAG retriever with semantic + lexical search.
    - Semantic: Top-k=12 from embeddings
    - Lexical: Top-k=12 from BM25
    - Merge & dedupe, rerank to top-6
    - Always includes federal/NERC unless explicitly excluded
    
    Supports three retrieval modes:
    - "hybrid": Semantic + BM25 (default)
    - "bm25": BM25 only (faster, exact keyword matches)
    - "semantic": Semantic only (natural language queries)
    """
    
    def __init__(
        self,
        use_reranker: bool = True,
        top_k_retrieve: int = 12,  # Increased from 8 for better recall
        top_k_rerank: int = 6,     # Increased from 4 for compliance
        default_mode: str = "hybrid",  # "hybrid", "bm25", "semantic"
    ):
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank
        self.use_reranker = use_reranker
        self.default_mode = default_mode
        
        # Load embedding model
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer(MODEL_NAME)
        
        # Load vector index (semantic)
        print("Loading vector index...")
        self.semantic_index = load_index()
        
        # Load BM25 index (lexical)
        print("Loading BM25 index...")
        self.bm25_index = load_bm25_index(self.semantic_index.chunks)
        
        # Load reranker if enabled
        self.reranker = None
        if use_reranker:
            print("Loading reranker...")
            self.reranker = CrossEncoder(RERANK_MODEL)
    
    def retrieve(
        self,
        query: str,
        states: list[str] | None = None,
        utility_domain: str = "electric",
        include_federal: bool = True,
        mode: str | None = None,  # Override default_mode for this query
    ) -> list[tuple[Chunk, float]]:
        """
        Retrieve chunks using specified mode.
        
        Args:
            query: Natural language query
            states: Filter to specific states (e.g., ["TX", "OK", "PA"])
            utility_domain: Filter by domain (default: "electric")
            include_federal: ALWAYS include federal/NERC (set to False to exclude)
            mode: Retrieval mode ("hybrid", "bm25", "semantic"). Defaults to self.default_mode.
        
        Returns:
            List of (Chunk, score) tuples, best first
        """
        mode = mode or self.default_mode
        
        if mode == "bm25":
            # BM25 only
            merged_list = self.bm25_index.search(
                query,
                top_k=self.top_k_retrieve,
                states=states,
                utility_domain=utility_domain,
                include_federal=include_federal,
            )
        elif mode == "semantic":
            # Semantic only
            query_emb = embed_query(query, self.embed_model)
            merged_list = self.semantic_index.search(
                query_emb,
                top_k=self.top_k_retrieve,
                states=states,
                utility_domain=utility_domain,
                include_federal=include_federal,
            )
        else:
            # Hybrid: semantic + lexical
            query_emb = embed_query(query, self.embed_model)
            semantic_results = self.semantic_index.search(
                query_emb,
                top_k=self.top_k_retrieve,
                states=states,
                utility_domain=utility_domain,
                include_federal=include_federal,
            )
            
            lexical_results = self.bm25_index.search(
                query,
                top_k=self.top_k_retrieve,
                states=states,
                utility_domain=utility_domain,
                include_federal=include_federal,
            )
            
            # Merge and dedupe by chunk_id
            merged = {}
            for chunk, score in semantic_results:
                merged[chunk.chunk_id] = (chunk, score)
            for chunk, score in lexical_results:
                if chunk.chunk_id not in merged:
                    merged[chunk.chunk_id] = (chunk, score)
            
            merged_list = list(merged.values())
        
        if not merged_list:
            return []
        
        # Rerank if enabled
        if self.use_reranker and self.reranker and len(merged_list) > self.top_k_rerank:
            # Prepare pairs for cross-encoder
            pairs = [(query, chunk.text) for chunk, _ in merged_list]
            
            # Score with cross-encoder
            rerank_scores = self.reranker.predict(pairs)
            
            # Sort by rerank score
            scored = list(zip(merged_list, rerank_scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            
            # Take top_k after reranking
            merged_list = [
                (chunk, float(rerank_score))
                for (chunk, _), rerank_score in scored[:self.top_k_rerank]
            ]
        
        return merged_list
    
    def format_context(self, results: list[tuple[Chunk, float]]) -> str:
        """
        Format retrieved chunks into context string for LLM.
        """
        context_parts = []
        
        for i, (chunk, score) in enumerate(results, 1):
            header = f"[Source {i}]"
            header += f" doc_id={chunk.doc_id}"
            header += f" state={chunk.state}"
            if chunk.page_start:
                if chunk.page_end and chunk.page_end != chunk.page_start:
                    header += f" pages={chunk.page_start}-{chunk.page_end}"
                else:
                    header += f" page={chunk.page_start}"
            if chunk.section:
                header += f" section=\"{chunk.section}\""
            
            context_parts.append(f"{header}\n{chunk.text}")
        
        return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    # Quick test
    retriever = Retriever(use_reranker=False)  # Skip reranker for faster test
    
    query = "What are the cybersecurity incident reporting requirements?"
    results = retriever.retrieve(query, states=["TX", "OK"])
    
    print(f"\nQuery: {query}")
    print(f"Results: {len(results)}")
    print("\n" + "="*60)
    
    for chunk, score in results[:3]:
        print(f"\n[{chunk.doc_id}] (score={score:.3f})")
        print(f"State: {chunk.state}, Pages: {chunk.page_start}-{chunk.page_end}")
        print(chunk.text[:300] + "...")
