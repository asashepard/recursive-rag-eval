"""
paragraph_index.py
------------------
Per-document paragraph-level indexing for RLM search_in_doc operations.
Stores paragraph spans with full provenance for deterministic retrieval.
"""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CORPUS_ROOT = Path(__file__).parent.parent / "corpus" / "normalized"
DOC_TEXT = CORPUS_ROOT / "doc_text"
DOC_META = CORPUS_ROOT / "doc_meta"
PARAGRAPHS_DIR = CORPUS_ROOT / "paragraphs"
SECTIONS_DIR = CORPUS_ROOT / "sections"


@dataclass
class ParagraphSpan:
    """A single paragraph with full provenance."""
    doc_id: str
    para_id: str
    page: int | None
    section_heading: str | None
    start_char: int
    end_char: int
    text: str
    
    def to_span_ref(self, text_preview_len: int = 200) -> dict:
        """Convert to span_ref format for RLM API. Uses span_id for efficient lookup."""
        preview = self.text[:text_preview_len]
        if len(self.text) > text_preview_len:
            preview += "..."
        return {
            "doc_id": self.doc_id,
            "span_id": self.para_id,  # Use para_id as span_id for get_paragraph lookup
            "page": self.page,
            "section_heading": self.section_heading,
            "text_preview": preview,
        }


@dataclass
class Section:
    """A document section with heading and location."""
    section_id: str
    heading: str
    level: int  # 1 = h1, 2 = h2, etc.
    page_start: int | None
    page_end: int | None
    start_char: int
    end_char: int


def extract_page_at_offset(text: str, offset: int) -> int | None:
    """Find page number at a given character offset."""
    pattern = re.compile(r"\[PAGE\s+(\d+)\]")
    page = None
    for m in pattern.finditer(text):
        if m.start() <= offset:
            page = int(m.group(1))
        else:
            break
    return page


def extract_section_at_offset(text: str, offset: int) -> str | None:
    """Find section heading at a given character offset."""
    pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
    section = None
    for m in pattern.finditer(text):
        if m.start() <= offset:
            section = m.group(2).strip()
        else:
            break
    return section


def split_into_paragraphs(text: str, min_len: int = 50) -> list[tuple[int, int, str]]:
    """
    Split text into paragraph units.
    Returns list of (start_char, end_char, text).
    """
    paragraphs = []
    
    # Split on double newlines (paragraph breaks)
    # Also split on [PAGE n] markers
    pattern = re.compile(r'\n\s*\n|\[PAGE\s+\d+\]')
    
    last_end = 0
    for match in pattern.finditer(text):
        start = last_end
        end = match.start()
        para_text = text[start:end].strip()
        
        if len(para_text) >= min_len:
            paragraphs.append((start, end, para_text))
        
        last_end = match.end()
    
    # Don't forget the last paragraph
    if last_end < len(text):
        para_text = text[last_end:].strip()
        if len(para_text) >= min_len:
            paragraphs.append((last_end, len(text), para_text))
    
    return paragraphs


def extract_sections(text: str) -> list[Section]:
    """Extract all section headings with their spans."""
    pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
    sections = []
    
    matches = list(pattern.finditer(text))
    
    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        start_char = m.start()
        
        # End is start of next section or end of document
        if i + 1 < len(matches):
            end_char = matches[i + 1].start()
        else:
            end_char = len(text)
        
        page_start = extract_page_at_offset(text, start_char)
        page_end = extract_page_at_offset(text, end_char)
        
        section = Section(
            section_id=f"sec_{i:03d}",
            heading=heading,
            level=level,
            page_start=page_start,
            page_end=page_end,
            start_char=start_char,
            end_char=end_char,
        )
        sections.append(section)
    
    return sections


def build_paragraph_index(doc_id: str, text: str) -> list[ParagraphSpan]:
    """Build paragraph spans for a single document."""
    paragraphs = split_into_paragraphs(text)
    spans = []
    
    for i, (start, end, para_text) in enumerate(paragraphs):
        page = extract_page_at_offset(text, start)
        section = extract_section_at_offset(text, start)
        
        span = ParagraphSpan(
            doc_id=doc_id,
            para_id=f"{doc_id}__para_{i:04d}",
            page=page,
            section_heading=section,
            start_char=start,
            end_char=end,
            text=para_text,
        )
        spans.append(span)
    
    return spans


def build_all_indexes():
    """Build paragraph indexes and section maps for all documents."""
    PARAGRAPHS_DIR.mkdir(exist_ok=True)
    SECTIONS_DIR.mkdir(exist_ok=True)
    
    all_paragraphs = []
    
    for meta_path in sorted(DOC_META.glob("*.json")):
        doc_id = meta_path.stem
        text_path = DOC_TEXT / f"{doc_id}.txt"
        
        if not text_path.exists():
            print(f"[SKIP] No text for {doc_id}")
            continue
        
        text = text_path.read_text(encoding="utf-8")
        
        # Build paragraph index
        paragraphs = build_paragraph_index(doc_id, text)
        all_paragraphs.extend(paragraphs)
        
        # Save per-document paragraphs
        para_path = PARAGRAPHS_DIR / f"{doc_id}.jsonl"
        with open(para_path, "w", encoding="utf-8") as f:
            for p in paragraphs:
                f.write(json.dumps(asdict(p)) + "\n")
        
        # Build section map
        sections = extract_sections(text)
        sec_path = SECTIONS_DIR / f"{doc_id}.json"
        with open(sec_path, "w", encoding="utf-8") as f:
            json.dump([asdict(s) for s in sections], f, indent=2)
        
        print(f"[OK] {doc_id}: {len(paragraphs)} paragraphs, {len(sections)} sections")
    
    # Save global paragraph index
    global_path = CORPUS_ROOT / "all_paragraphs.jsonl"
    with open(global_path, "w", encoding="utf-8") as f:
        for p in all_paragraphs:
            f.write(json.dumps(asdict(p)) + "\n")
    
    print(f"\nTotal: {len(all_paragraphs)} paragraphs across all documents")
    return all_paragraphs


class DocumentParagraphIndex:
    """Per-document BM25 index over paragraphs."""
    
    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.paragraphs: list[ParagraphSpan] = []
        self.bm25: BM25Okapi | None = None
        self._load()
    
    def _load(self):
        """Load paragraphs and build BM25."""
        para_path = PARAGRAPHS_DIR / f"{self.doc_id}.jsonl"
        if not para_path.exists():
            return
        
        with open(para_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.paragraphs.append(ParagraphSpan(**data))
        
        if self.paragraphs:
            tokenized = [p.text.lower().split() for p in self.paragraphs]
            self.bm25 = BM25Okapi(tokenized)
    
    def search(self, query: str, k: int = 10) -> list[ParagraphSpan]:
        """Search within this document."""
        if not self.bm25:
            return []
        
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k by score
        scored = [(p, s) for p, s in zip(self.paragraphs, scores) if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [p for p, _ in scored[:k]]


class GlobalParagraphIndex:
    """Global BM25 index over all paragraphs."""
    
    def __init__(self):
        self.paragraphs: list[ParagraphSpan] = []
        self.bm25: BM25Okapi | None = None
        self._load()
    
    def _load(self):
        """Load all paragraphs and build BM25."""
        global_path = CORPUS_ROOT / "all_paragraphs.jsonl"
        if not global_path.exists():
            return
        
        with open(global_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.paragraphs.append(ParagraphSpan(**data))
        
        if self.paragraphs:
            tokenized = [p.text.lower().split() for p in self.paragraphs]
            self.bm25 = BM25Okapi(tokenized)
        
        print(f"Loaded global paragraph index: {len(self.paragraphs)} paragraphs")
    
    def search(
        self,
        query: str,
        k: int = 20,
        states: list[str] | None = None,
        include_federal: bool = True,
    ) -> list[ParagraphSpan]:
        """Search across all paragraphs with optional state filter."""
        if not self.bm25:
            return []
        
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        
        # Build allowed states set
        allowed_states = None
        if states is not None:
            allowed_states = set(states)
            if include_federal:
                allowed_states.add("FED")
                allowed_states.add("NARUC")
        
        # Filter and score
        results = []
        for para, score in zip(self.paragraphs, scores):
            if score <= 0:
                continue
            
            if allowed_states is not None:
                # Extract state from doc_id (first part before _)
                doc_state = para.doc_id.split("_")[0]
                if doc_state not in allowed_states:
                    continue
            
            results.append((para, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in results[:k]]
    
    def get_paragraph(self, para_id: str) -> ParagraphSpan | None:
        """Retrieve a paragraph by its ID (span_id) for efficient lookup."""
        for para in self.paragraphs:
            if para.para_id == para_id:
                return para
        return None


if __name__ == "__main__":
    build_all_indexes()
