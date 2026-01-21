"""
chunker.py
----------
Chunk normalized corpus text into passages for embedding.
Stores metadata per chunk for citation reconstruction.
"""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import tiktoken

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CORPUS_ROOT = Path(__file__).resolve().parent.parent / "corpus" / "normalized"
DOC_TEXT = CORPUS_ROOT / "doc_text"
DOC_META = CORPUS_ROOT / "doc_meta"
CHUNKS_OUT = CORPUS_ROOT / "chunks.jsonl"

# Chunking params
CHUNK_SIZE = 900       # target tokens per chunk
CHUNK_OVERLAP = 150    # overlap tokens
MIN_CHUNK_SIZE = 200   # don't create tiny trailing chunks

# Tokenizer (cl100k_base works for GPT-4, similar to most embedding models)
ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    """A single chunk with full provenance metadata."""
    chunk_id: str
    doc_id: str
    state: str
    jurisdiction: str
    utility_domain: str
    source_type: str
    url: str
    page_start: int | None
    page_end: int | None
    section: str | None
    section_heading: str | None
    clause_id: str | None  # e.g., "CIP-008-6 R1" or "Chapter 101 §3025"
    char_start: int
    char_end: int
    token_count: int
    text: str


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


def extract_page_markers(text: str) -> list[tuple[int, int]]:
    """
    Find [PAGE n] markers and return list of (page_num, char_offset).
    """
    pattern = re.compile(r"\[PAGE\s+(\d+)\]")
    markers = []
    for m in pattern.finditer(text):
        markers.append((int(m.group(1)), m.start()))
    return markers


def get_page_at_offset(markers: list[tuple[int, int]], offset: int) -> int | None:
    """Given char offset, return which page it falls on."""
    if not markers:
        return None
    page = None
    for pnum, poff in markers:
        if poff <= offset:
            page = pnum
        else:
            break
    return page


def extract_section_at_offset(text: str, offset: int) -> str | None:
    """
    Find the nearest preceding heading (markdown-style # or ##).
    """
    pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
    section = None
    for m in pattern.finditer(text):
        if m.start() <= offset:
            section = m.group(2).strip()
        else:
            break
    return section


def extract_clause_id_at_offset(text: str, offset: int) -> str | None:
    """
    Extract regulatory reference like:
    - CIP-008-6 R1, R1.1, etc.
    - Chapter 101 §3025
    - 165:35-33-7
    """
    # Patterns for common regulatory citations
    patterns = [
        r"(CIP-\d{3}-\d+\s+(?:R|Requirement)\s+\d+(?:\.\d+)*)",
        r"(Chapter\s+\d+\s+§\s*\d+(?:\.\d+)*)",
        r"(\d+:\d+-\d+-\d+(?:\.\d+)*)",
    ]
    
    clause = None
    for pattern_str in patterns:
        pattern = re.compile(pattern_str)
        for m in pattern.finditer(text):
            if m.start() <= offset:
                clause = m.group(1)
            else:
                break
    return clause


def chunk_document(doc_id: str, text: str, meta: dict) -> list[Chunk]:
    """
    Split a document into overlapping chunks.
    Respects section boundaries to avoid cutting obligations across chunks.
    """
    chunks = []
    page_markers = extract_page_markers(text)
    
    tokens = ENC.encode(text)
    total_tokens = len(tokens)
    
    idx = 0
    chunk_num = 0
    
    while idx < total_tokens:
        # Determine end of this chunk
        end_idx = min(idx + CHUNK_SIZE, total_tokens)
        
        # Decode chunk text
        chunk_tokens = tokens[idx:end_idx]
        chunk_text = ENC.decode(chunk_tokens)
        
        # Calculate character offsets (approximate via ratio)
        char_ratio = len(text) / total_tokens if total_tokens else 1
        char_start = int(idx * char_ratio)
        char_end = int(end_idx * char_ratio)
        
        # Get page range
        page_start = get_page_at_offset(page_markers, char_start)
        page_end = get_page_at_offset(page_markers, char_end)
        
        # Get section heading
        section = extract_section_at_offset(text, char_start)
        
        # Get clause ID (regulatory reference)
        clause = extract_clause_id_at_offset(text, char_start)
        
        chunk = Chunk(
            chunk_id=f"{doc_id}__chunk_{chunk_num:03d}",
            doc_id=doc_id,
            state=meta.get("state", "UNK"),
            jurisdiction=meta.get("jurisdiction", "unknown"),
            utility_domain=meta.get("utility_domain", "unknown"),
            source_type=meta.get("source_type", "unknown"),
            url=meta.get("url", ""),
            page_start=page_start,
            page_end=page_end,
            section=section,
            section_heading=section,
            clause_id=clause,
            char_start=char_start,
            char_end=char_end,
            token_count=len(chunk_tokens),
            text=chunk_text,
        )
        chunks.append(chunk)
        
        # Move forward (with overlap)
        idx += CHUNK_SIZE - CHUNK_OVERLAP
        chunk_num += 1
        
        # Avoid tiny trailing chunks
        remaining = total_tokens - idx
        if 0 < remaining < MIN_CHUNK_SIZE:
            # Extend the last chunk instead
            break
    
    return chunks


def build_chunks() -> list[Chunk]:
    """
    Process all documents and build chunk corpus.
    """
    all_chunks = []
    
    for meta_path in sorted(DOC_META.glob("*.json")):
        doc_id = meta_path.stem
        text_path = DOC_TEXT / f"{doc_id}.txt"
        
        if not text_path.exists():
            print(f"[SKIP] No text file for {doc_id}")
            continue
        
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        text = text_path.read_text(encoding="utf-8")
        
        if not text.strip():
            print(f"[SKIP] Empty text for {doc_id}")
            continue
        
        chunks = chunk_document(doc_id, text, meta)
        all_chunks.extend(chunks)
        print(f"[OK] {doc_id}: {len(chunks)} chunks")
    
    return all_chunks


def save_chunks(chunks: list[Chunk], out_path: Path = CHUNKS_OUT):
    """Write chunks to JSONL."""
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk)) + "\n")
    print(f"\nSaved {len(chunks)} chunks to {out_path}")


def load_chunks(path: Path = CHUNKS_OUT) -> list[Chunk]:
    """Load chunks from JSONL."""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            chunks.append(Chunk(**data))
    return chunks


if __name__ == "__main__":
    chunks = build_chunks()
    save_chunks(chunks)
