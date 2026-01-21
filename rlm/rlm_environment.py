"""
rlm_environment.py
------------------
RLM (Recursive Language Model) environment API.
Pure Python tools for document exploration - no LLM inside tools.
The controller (LLM agent) calls these to gather evidence.
"""

import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# Add parent for rag imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rlm.paragraph_index import (
    Section,
    DocumentParagraphIndex,
    GlobalParagraphIndex,
    DOC_TEXT,
    DOC_META,
    SECTIONS_DIR,
)
from rag.retriever import Retriever

# ---------------------------------------------------------------------------
# Span Reference (standard format for all tools)
# ---------------------------------------------------------------------------

@dataclass
class SpanRef:
    """Standard reference to a text span in the corpus."""
    doc_id: str
    page: int | None
    section_heading: str | None
    start_char: int
    end_char: int
    text_preview: str
    
    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Document Info
# ---------------------------------------------------------------------------

@dataclass
class DocInfo:
    """Document metadata with section headings."""
    doc_id: str
    title: str
    state: str
    jurisdiction: str
    url: str
    source_type: str
    headings: list[dict]  # [{section_id, heading, level, page_start}]


# ---------------------------------------------------------------------------
# RLM Environment (Tool Implementations)
# ---------------------------------------------------------------------------

class RLMEnvironment:
    """
    RLM environment with document exploration tools.
    All tools are pure Python - no LLM calls.
    
    Supports three retrieval modes:
    - "hybrid": Semantic + BM25 (default, best for most queries)
    - "bm25": BM25 only (faster, better for exact keyword matches)
    - "semantic": Semantic only (better for natural language queries)
    """
    
    def __init__(self, retrieval_mode: str = "hybrid"):
        self.retrieval_mode = retrieval_mode
        
        # Global paragraph index (for global_search)
        print("Loading global paragraph index...")
        self.global_para_index = GlobalParagraphIndex()
        
        # Hybrid retriever (for retrieve_chunks)
        print("Loading hybrid retriever...")
        self.retriever = Retriever(
            use_reranker=False,  # Faster for exploration
            default_mode=retrieval_mode,
        )
        
        # Per-document indexes (lazy loaded)
        self._doc_indexes: dict[str, DocumentParagraphIndex] = {}
        
        # Document metadata cache
        self._doc_meta: dict[str, dict] = {}
        self._load_doc_meta()
        
        # Document text cache (lazy loaded)
        self._doc_texts: dict[str, str] = {}
    
    def _load_doc_meta(self):
        """Load all document metadata."""
        for meta_path in DOC_META.glob("*.json"):
            doc_id = meta_path.stem
            with open(meta_path, "r", encoding="utf-8") as f:
                self._doc_meta[doc_id] = json.load(f)
    
    def _get_doc_index(self, doc_id: str) -> DocumentParagraphIndex:
        """Lazy load per-document index."""
        if doc_id not in self._doc_indexes:
            self._doc_indexes[doc_id] = DocumentParagraphIndex(doc_id)
        return self._doc_indexes[doc_id]
    
    def _get_doc_text(self, doc_id: str) -> str:
        """Lazy load document text."""
        if doc_id not in self._doc_texts:
            text_path = DOC_TEXT / f"{doc_id}.txt"
            if text_path.exists():
                self._doc_texts[doc_id] = text_path.read_text(encoding="utf-8")
            else:
                self._doc_texts[doc_id] = ""
        return self._doc_texts[doc_id]
    
    def _get_sections(self, doc_id: str) -> list[Section]:
        """Load sections for a document."""
        sec_path = SECTIONS_DIR / f"{doc_id}.json"
        if not sec_path.exists():
            return []
        
        with open(sec_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return [Section(**s) for s in data]
    
    # -----------------------------------------------------------------------
    # Global Tools
    # -----------------------------------------------------------------------
    
    def global_search(
        self,
        query: str,
        states: list[str] | None = None,
        include_federal: bool = True,
        k: int = 10,
    ) -> list[dict]:
        """
        BM25 search across all paragraph spans.
        Returns span_refs with span_id for efficient lookup via get_paragraph.
        
        Args:
            query: Search query (keywords work well)
            states: Filter to these states (e.g., ["TX", "OK"])
            include_federal: Include FED/NARUC docs
            k: Max results (capped at 12 for efficiency)
        
        Returns:
            List of span_ref dicts with span_id field
        """
        results = self.global_para_index.search(
            query=query,
            k=min(k, 12),  # Cap at 12 for budget control
            states=states,
            include_federal=include_federal,
        )
        
        return [p.to_span_ref() for p in results]
    
    def retrieve_chunks(
        self,
        query: str,
        states: list[str] | None = None,
        utility_domain: str = "electric",
        include_federal: bool = True,
        k: int = 12,
    ) -> list[dict]:
        """
        Hybrid semantic+BM25 retrieval over 900-token chunks.
        Use this for broad context before drilling down.
        
        Args:
            query: Natural language query
            states: Filter to these states
            utility_domain: "electric", "gas", etc.
            include_federal: Include federal docs
            k: Max results
        
        Returns:
            List of chunk references
        """
        results = self.retriever.retrieve(
            query=query,
            states=states,
            utility_domain=utility_domain,
            include_federal=include_federal,
        )
        
        # Convert chunks to span_ref-like format
        refs = []
        for chunk, score in results[:k]:
            refs.append({
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "page": chunk.page_start,
                "page_end": chunk.page_end,
                "section_heading": chunk.section_heading,
                "start_char": chunk.char_start,
                "end_char": chunk.char_end,
                "text_preview": chunk.text[:150] + "..." if len(chunk.text) > 150 else chunk.text,
                "score": score,
            })
        
        return refs
    
    def open_doc(self, doc_id: str) -> Optional[dict]:
        """
        Open a document and get its metadata + section headings.
        
        Args:
            doc_id: Document identifier (e.g., "TX_ERCOT_NPRR928")
        
        Returns:
            DocInfo dict with title, state, url, headings
        """
        if doc_id not in self._doc_meta:
            return None
        
        meta = self._doc_meta[doc_id]
        sections = self._get_sections(doc_id)
        
        headings = [
            {
                "section_id": s.section_id,
                "heading": s.heading,
                "level": s.level,
                "page_start": s.page_start,
            }
            for s in sections
        ]
        
        return {
            "doc_id": doc_id,
            "title": meta.get("title", doc_id),
            "state": meta.get("state", "UNK"),
            "jurisdiction": meta.get("jurisdiction", "unknown"),
            "url": meta.get("url", ""),
            "source_type": meta.get("source_type", "unknown"),
            "headings": headings,
        }
    
    def get_span(
        self,
        doc_id: str,
        start_char: int,
        end_char: int,
    ) -> Optional[str]:
        """
        Get raw text for a character span.
        
        Args:
            doc_id: Document identifier
            start_char: Start character offset
            end_char: End character offset
        
        Returns:
            Text content of the span
        """
        text = self._get_doc_text(doc_id)
        if not text:
            return None
        
        # Clamp to valid range
        start = max(0, start_char)
        end = min(len(text), end_char)
        
        return text[start:end]
    
    def get_page(self, doc_id: str, page_num: int) -> Optional[str]:
        """
        Get all text from a specific page.
        
        Args:
            doc_id: Document identifier
            page_num: Page number (1-indexed)
        
        Returns:
            Text content of the page
        """
        text = self._get_doc_text(doc_id)
        if not text:
            return None
        
        # Find page markers
        pattern = re.compile(r"\[PAGE\s+(\d+)\]")
        markers = [(int(m.group(1)), m.start(), m.end()) for m in pattern.finditer(text)]
        
        if not markers:
            return None
        
        # Find this page
        for i, (pnum, start, end) in enumerate(markers):
            if pnum == page_num:
                # Start after [PAGE n] marker
                page_start = end
                
                # End at next page or end of doc
                if i + 1 < len(markers):
                    page_end = markers[i + 1][1]
                else:
                    page_end = len(text)
                
                return text[page_start:page_end].strip()
        
        return None
    
    # -----------------------------------------------------------------------
    # In-Document Tools
    # -----------------------------------------------------------------------
    
    def search_in_doc(
        self,
        doc_id: str,
        query: str,
        k: int = 8,
    ) -> list[dict]:
        """
        BM25 search within a single document.
        Use after open_doc to find specific clauses.
        Returns span_refs with span_id for get_paragraph lookup.
        
        Args:
            doc_id: Document to search in
            query: Search query
            k: Max results (default 8)
        
        Returns:
            List of span_ref dicts with span_id
        """
        index = self._get_doc_index(doc_id)
        results = index.search(query, k=k)
        
        return [p.to_span_ref() for p in results]
    
    def get_paragraph(self, span_id: str) -> dict | None:
        """
        Retrieve full paragraph text by span_id.
        Use after global_search or search_in_doc to get complete text.
        
        Args:
            span_id: The span_id from a search result (e.g., 'TX_ERCOT_NPRR928__para_0042')
        
        Returns:
            Dict with doc_id, span_id, page, section_heading, text (full paragraph)
        """
        para = self.global_para_index.get_paragraph(span_id)
        if not para:
            return None
        
        return {
            "doc_id": para.doc_id,
            "span_id": para.para_id,
            "page": para.page,
            "section_heading": para.section_heading,
            "text": para.text,  # Full text, not preview
        }
    
    def follow_reference(
        self,
        doc_id: str,
        ref_string: str,
    ) -> Optional[dict]:
        """
        Follow a regulatory reference within a document.
        Handles: "§3025", "R1.1", "Rule 165:35-33-7", "Chapter 101", etc.
        
        Args:
            doc_id: Document to search in
            ref_string: Reference string (e.g., "R1.1", "§3025")
        
        Returns:
            span_ref dict if found, None otherwise
        """
        text = self._get_doc_text(doc_id)
        if not text:
            return None
        
        # Build search patterns for common reference formats
        patterns = [
            # Section symbol: §3025 or § 3025
            rf"§\s*{re.escape(ref_string.lstrip('§').strip())}",
            # CIP requirements: R1, R1.1, R1.1.1
            rf"(?:Requirement\s+)?{re.escape(ref_string)}(?:\s|\.|\)|,)",
            # Rule references: 165:35-33-7
            rf"(?:Rule\s+)?{re.escape(ref_string)}",
            # Chapter references: Chapter 101
            rf"Chapter\s+{re.escape(ref_string.replace('Chapter', '').strip())}",
            # Exact match
            re.escape(ref_string),
        ]
        
        for pattern_str in patterns:
            try:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                match = pattern.search(text)
                if match:
                    # Get surrounding context (500 chars before and after)
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 400)
                    
                    # Find page at this location
                    page = None
                    page_pattern = re.compile(r"\[PAGE\s+(\d+)\]")
                    for m in page_pattern.finditer(text):
                        if m.start() <= match.start():
                            page = int(m.group(1))
                        else:
                            break
                    
                    # Find section at this location
                    section = None
                    sec_pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
                    for m in sec_pattern.finditer(text):
                        if m.start() <= match.start():
                            section = m.group(2).strip()
                        else:
                            break
                    
                    span_text = text[start:end]
                    
                    return {
                        "doc_id": doc_id,
                        "page": page,
                        "section_heading": section,
                        "start_char": start,
                        "end_char": end,
                        "text_preview": span_text[:200] + "..." if len(span_text) > 200 else span_text,
                        "matched_ref": ref_string,
                    }
            except re.error:
                continue
        
        return None
    
    # -----------------------------------------------------------------------
    # Utility Methods
    # -----------------------------------------------------------------------
    
    def list_documents(
        self,
        states: list[str] | None = None,
    ) -> list[dict]:
        """List all documents, optionally filtered by state."""
        docs = []
        
        for doc_id, meta in self._doc_meta.items():
            state = meta.get("state", "UNK")
            
            if states is not None:
                allowed = set(states)
                allowed.add("FED")
                allowed.add("NARUC")
                if state not in allowed:
                    continue
            
            docs.append({
                "doc_id": doc_id,
                "title": meta.get("title", doc_id),
                "state": state,
                "source_type": meta.get("source_type", "unknown"),
            })
        
        return sorted(docs, key=lambda x: (x["state"], x["doc_id"]))
    
    def available_states(self) -> list[str]:
        """Get list of states in corpus."""
        states = set()
        for meta in self._doc_meta.values():
            states.add(meta.get("state", "UNK"))
        return sorted(states)


# ---------------------------------------------------------------------------
# Tool Definitions for LLM Controller
# ---------------------------------------------------------------------------

RLM_TOOLS = [
    {
        "name": "global_search",
        "description": "BM25 search across all documents. Returns span_refs with span_id for get_paragraph lookup. Use for finding clauses with specific keywords.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords to search for"},
                "states": {"type": "array", "items": {"type": "string"}, "description": "Filter to specific states (e.g., ['TX', 'OK'])"},
                "k": {"type": "integer", "description": "Max results (default 10, capped at 12)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "retrieve_chunks",
        "description": "Hybrid semantic+keyword retrieval over larger chunks. Use for broad document discovery.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language query"},
                "states": {"type": "array", "items": {"type": "string"}, "description": "Filter to specific states"},
                "k": {"type": "integer", "description": "Max results (default 12)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "open_doc",
        "description": "Open a document to see its metadata and section headings.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "Document identifier (e.g., 'TX_ERCOT_NPRR928')"},
            },
            "required": ["doc_id"],
        },
    },
    {
        "name": "get_paragraph",
        "description": "Get full paragraph text by span_id. Use after search to read complete text.",
        "parameters": {
            "type": "object",
            "properties": {
                "span_id": {"type": "string", "description": "The span_id from search results (e.g., 'TX_ERCOT__para_0042')"},
            },
            "required": ["span_id"],
        },
    },
    {
        "name": "get_page",
        "description": "Get all text from a specific page. Use for PDFs when you know the page number.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "Document identifier"},
                "page_num": {"type": "integer", "description": "Page number (1-indexed)"},
            },
            "required": ["doc_id", "page_num"],
        },
    },
    {
        "name": "search_in_doc",
        "description": "BM25 search within a single document. Returns span_refs with span_id for get_paragraph lookup.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "Document to search in"},
                "query": {"type": "string", "description": "Keywords to search for"},
                "k": {"type": "integer", "description": "Max results (default 8)"},
            },
            "required": ["doc_id", "query"],
        },
    },
    {
        "name": "follow_reference",
        "description": "Follow a regulatory reference within a document. Handles section, R1.1, Rule numbers, Chapter refs.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "Document to search in"},
                "ref_string": {"type": "string", "description": "Reference (e.g., 'R1.1', 'Chapter 101')"},
            },
            "required": ["doc_id", "ref_string"],
        },
    },
]


if __name__ == "__main__":
    # Quick test
    env = RLMEnvironment()
    
    print("\n=== Available States ===")
    print(env.available_states())
    
    print("\n=== Global Search: 'shall notify' ===")
    results = env.global_search("shall notify", states=["TX", "FED"], k=3)
    for r in results:
        print(f"  {r['doc_id']}: {r['text_preview'][:80]}...")
    
    print("\n=== Open Doc: TX_ERCOT_NPRR928 ===")
    doc = env.open_doc("TX_ERCOT_NPRR928")
    if doc:
        print(f"  Title: {doc['title']}")
        print(f"  Headings: {len(doc['headings'])}")
    
    print("\n=== Search in Doc ===")
    results = env.search_in_doc("FED_NERC_CIP-008-6", "reportable incident", k=2)
    for r in results:
        print(f"  Page {r['page']}: {r['text_preview'][:80]}...")
