# System Architecture

**Technical reference for the RAG and Controller-Driven extraction systems.**

This document covers implementation details, code structure, and component interactions. For running experiments and interpreting results, see [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md).

> **Terminology:** The README uses "Controller-Driven" to describe the iterative approach. In code and configs, this is `rlm`. Both terms refer to the same system.

---

## Overview

**Purpose:** Compare two evidence consumption strategies for regulatory compliance extraction.

Both systems answer queries like *"What are the incident reporting requirements for New Jersey?"* by:

1. Searching a corpus of 17 regulatory documents
2. Extracting structured obligations using an LLM
3. Returning JSON with citations

They differ in **how** they consume evidence: single-pass vs. iterative.

### Federal Baseline Policy

State queries return both state AND federal obligations. Federal rules (NERC CIP, DOE) apply to all utilities.

See [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md#federal-baseline-policy) for scoring details.

---

## Corpus Structure

```
corpus/normalized/
├── doc_text/           # Plain text per document (17 files)
├── doc_meta/           # Metadata JSON per document
├── chunks.jsonl        # 900-token chunks for RAG
├── embeddings.npy      # BGE-large vectors (384 dims)
├── faiss.index         # Semantic search index
├── bm25_index.json     # Keyword search index
└── paragraphs/         # Per-document paragraph indexes
```

### Document Processing

**Chunking:** Documents are split into ~900-token overlapping chunks by [rag/chunker.py](rag/chunker.py). Each chunk preserves metadata (doc_id, state, page number).

**Embeddings:** Chunks are embedded using `BAAI/bge-large-en-v1.5` in [rag/embeddings.py](rag/embeddings.py). The model is loaded via `SentenceTransformer`.

**Indexes:**
- **FAISS:** Built in [rag/index.py](rag/index.py) for fast approximate nearest neighbor search
- **BM25:** Built in [rag/bm25_search.py](rag/bm25_search.py) using `rank_bm25.BM25Okapi`
- **Paragraph:** Fine-grained per-document indexes in [rlm/paragraph_index.py](rlm/paragraph_index.py)

---

## RAG System

> Code config name: `rag`

### Architecture

```
Query → Retriever → top-6 chunks → LLM extraction → ActivityResponse
```

Single-pass: one retrieval, one LLM call.

### Components

#### Retriever ([rag/retriever.py](rag/retriever.py))

Hybrid semantic + lexical retrieval:

```python
class Retriever:
    def retrieve(self, query, states, include_federal):
        # Semantic: embed query, search FAISS
        semantic_results = self.semantic_index.search(query_emb, top_k=12)
        
        # Lexical: BM25 keyword search
        lexical_results = self.bm25_index.search(query, top_k=12)
        
        # Merge, dedupe, rerank with cross-encoder
        return self.reranker.predict(merged)[:6]
```

Key parameters:
- `top_k_retrieve=12`: Initial candidates from each index
- `top_k_rerank=6`: Final results after cross-encoder reranking
- `RERANK_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"`

#### RAGRunner ([eval/rag_runner.py](eval/rag_runner.py))

Orchestrates retrieval and extraction:

```python
class RAGRunner:
    def research(self, activity, states):
        # 1. Retrieve chunks (includes both state and federal docs)
        results = self.retriever.retrieve(query_text, states)
        
        # 2. Build context from chunks
        context = self._format_chunks(results)
        
        # 3. Single LLM call with unified prompt
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": EXTRACTION_USER_TEMPLATE.format(...)},
            ],
            response_format={"type": "json_object"},
        )
        
        # 4. Parse and return
        return ActivityResponse(...)
```

**No Pre-LLM Gating:** The LLM is always called when retrieval returns chunks. Negative test behavior relies solely on prompt instructions ("return empty array if no obligations found").

---

## Controller-Driven System

> Code config name: `rlm`

### Architecture

```
Query → Controller → [discover → search → extract → verify → repair] × N → ActivityResponse
```

Iterative: multiple search and extraction steps per query, with post-extraction verification.

### Controller ([rlm/rlm_controller.py](rlm/rlm_controller.py))

The main orchestration loop:

```python
class RLMController:
    def research(self, activity, states):
        for state in states:
            # Phase 1: Discover documents
            doc_ids = self._discover_documents(activity, state)
            
            # Phase 2: Extract from each document
            for doc_id in doc_ids[:MAX_DOCS_PER_STATE]:
                obligations = self._extract_from_document(doc_id, activity, state)
                state_results.extend(obligations)
        
        return self._build_response(activity, state_results)
```

#### Budget Controls

Hardcoded limits prevent runaway execution:

```python
MAX_ITERATIONS_GLOBAL = 12      # Total operations cap
MAX_TOOL_CALLS_PER_STATE = 6    # Per-state tool calls
MAX_DOCS_PER_STATE = 2          # Documents to explore per state
MAX_SPANS_PER_STATE = 10        # Paragraphs to read per state (increased for better recall)
```

#### Document Discovery

Uses BM25 search filtered by state:

```python
def _discover_documents(self, activity, state):
    results = self.env.global_search(
        query=f"{activity} {state}",
        states=[state],
        include_federal=False,
    )
    # Return unique doc_ids from top results
    return list(dict.fromkeys([r["doc_id"] for r in results]))
```

#### Query Expansion (LLM-Based)

Before searching within documents, queries are expanded using the LLM to generate domain-aware variants:

```python
def _expand_query(self, activity: str) -> list[str]:
    """Use LLM to generate domain-aware query variants for BM25 search."""
    queries = [activity]
    
    expansion_prompt = f"""Generate 6 alternative search phrases for: "{activity}"
    Include synonyms and legal/regulatory jargon...
    Return JSON: {{"expansions": [...]}}"""
    
    response = self.client.chat.completions.create(
        model=self.model,
        temperature=0,  # Deterministic for reproducibility
        ...
    )
    
    # Returns: ["incident reporting", "event notification", "breach disclosure", ...]
    return queries[:5]  # Original + up to 4 expansions
```

**Example expansion for "incident reporting":**
```json
{"expansions": ["event notification", "breach disclosure", "cyber incident notice", "security occurrence report", "reportable event", "incident notification"]}
```

> **Fairness Note:** RAG's hybrid retrieval uses semantic embeddings, which *implicitly* capture synonym relationships ("notification" ≈ "report" in embedding space). Controller-Driven's within-document BM25 search has no semantic layer, so LLM-based expansion achieves parity. The LLM can also generate domain-specific jargon (e.g., "Reportable Cyber Security Incident" from NERC) that frozen embeddings may not capture.

#### Contextual Windowing

Regulatory obligations sometimes span paragraph boundaries. When extracting, the controller includes the preceding paragraph:

```python
def _get_context_window(self, span_id):
    """Return current paragraph plus preceding context."""
    current = self.env.get_paragraph(span_id)
    prev_id = self._get_previous_paragraph_id(span_id)
    if prev_id:
        prev = self.env.get_paragraph(prev_id)
        return prev["text"] + "\n\n" + current["text"]
    return current["text"]
```

> **Fairness Note:** RAG uses 900-token overlapping chunks, which naturally provide cross-boundary context. Controller-Driven's paragraph-level granularity requires explicit windowing to achieve comparable context coverage.

#### Paragraph Extraction

For each document, searches for relevant paragraphs and extracts from each:

```python
def _extract_from_document(self, doc_id, activity, state):
    # Expand query with synonyms
    expanded_query = self._expand_query(activity)
    
    # Search within document (k=25 for better recall)
    spans = self.env.search_in_doc(doc_id, expanded_query, k=25)
    
    for span in spans:
        # Get full paragraph text for filtering
        full_span = self.env.get_paragraph(span["span_id"])
        
        # Pre-filter on FULL text, not just preview
        if not self._has_obligation_language(full_span["text"]):
            continue
        
        # Include preceding paragraph for context
        context_text = self._get_context_window(span["span_id"])
        
        # LLM extraction returns ALL obligations in paragraph
        extracted = self._extract_obligation_from_span(context_text, doc_id, activity, state)
        
        for obligation in extracted:  # May be multiple
            if not self._is_relevant_obligation(obligation, activity):
                continue
            obligations.append(obligation)
```

#### Multi-Obligation Extraction

A single paragraph may contain multiple obligations (e.g., different reporting timelines). The extractor now returns a list:

```python
def _extract_obligation_from_span(self, text, doc_id, activity, state) -> list[Obligation]:
    """Extract ALL obligations from a span, not just the first."""
    # LLM call returns {"obligations": [...]}
    result = self._call_llm(text, activity, state)
    return [Obligation(**ob) for ob in result.get("obligations", [])]
```

#### Verification & Repair Loop

After extraction, each obligation is verified and optionally repaired:

```python
def _verify_and_repair(self, obligation, paragraph_text, cost):
    # Step 1: Verify the obligation
    verification = self._call_verifier(obligation, paragraph_text)
    
    # If activity doesn't match, drop the obligation
    if not verification.get("activity_match"):
        cost.rlm_obligations_dropped_by_verifier += 1
        return None
    
    cost.rlm_obligations_verified += 1
    
    # Step 2: Null out unsupported fields
    if not verification.get("deadline_present"):
        obligation.deadline = None
    if not verification.get("notify_present"):
        obligation.notify_who = None
    
    # Step 3: Attempt repair for missing fields
    if obligation.deadline is None or obligation.notify_who is None:
        repair = self._call_repair(obligation, paragraph_text)
        cost.rlm_repair_attempts += 1
        
        # Accept only if substring appears verbatim in paragraph
        if repair.get("deadline") and obligation.deadline is None:
            if self._validate_substring(repair["deadline"], paragraph_text):
                obligation.deadline = repair["deadline"]
                cost.rlm_fields_repaired_deadline += 1
            else:
                cost.rlm_repair_rejected_deadline += 1
        
        # Same for notify_who...
    
    return obligation
```

The `_validate_substring` check ensures that any repaired field value actually appears as a substring in the source paragraph, preventing hallucinated repairs.

### Environment ([rlm/rlm_environment.py](rlm/rlm_environment.py))

Pure Python tools (no LLM calls) for document exploration:

```python
class RLMEnvironment:
    def global_search(self, query, states, k):
        """BM25 search across all paragraphs."""
        return self.global_para_index.search(query, k, states)
    
    def search_in_doc(self, doc_id, query, k):
        """BM25 search within a single document."""
        doc_index = self._get_doc_index(doc_id)
        return doc_index.search(query, k)
    
    def get_paragraph(self, span_id):
        """Retrieve full text for a paragraph span."""
        return self.global_para_index.get_by_id(span_id)
    
    def retrieve_chunks(self, query, states, k):
        """Hybrid semantic+BM25 over 900-token chunks."""
        return self.retriever.retrieve(query, states)
```

### Paragraph Index ([rlm/paragraph_index.py](rlm/paragraph_index.py))

Fine-grained indexing at paragraph level:

```python
@dataclass
class ParagraphSpan:
    doc_id: str
    para_id: str
    page: int | None
    section_heading: str | None
    start_char: int
    end_char: int
    text: str

class GlobalParagraphIndex:
    """BM25 index over all 1020 paragraphs."""
    def search(self, query, k, states):
        scores = self.bm25.get_scores(tokenize(query))
        # Filter by state, return top-k
        ...
```

---

## Shared Components

### Output Schema ([rag/schemas.py](rag/schemas.py))

Both systems produce the same Pydantic models:

```python
class Obligation(BaseModel):
    obligation: str
    trigger: Optional[str]
    deadline: Optional[str]
    notify_who: Optional[str]
    required_content: Optional[str]
    citations: list[Citation]

class StateObligations(BaseModel):
    obligations: list[Obligation]
    confidence: str  # "high", "partial", "low"
    not_found_explanation: Optional[str]

class ActivityResponse(BaseModel):
    activity: str
    states: dict[str, StateObligations]
    federal_baseline: Optional[StateObligations]
```

### Extraction Prompts ([eval/experiment_config.py](eval/experiment_config.py))

Unified prompts used by both RAG and Controller-Driven:

```python
EXTRACTION_SYSTEM_PROMPT = """You are a regulatory compliance expert...

RULES:
1. Only extract obligations EXPLICITLY stated in the provided text.
2. If no obligations, return {"obligations": []}.
3. Every obligation MUST include a citation with doc_id and quote.
4. Every non-null field must be supported by a citation quote.
5. Do not use information from outside the provided text.
"""

EXTRACTION_USER_TEMPLATE = """QUERY: What are the {activity} requirements for {state}?

REGULATORY TEXT:
{context}

Extract all compliance obligations..."""
```

### Keywords and Filters ([eval/keywords.py](eval/keywords.py))

Regex patterns for filtering and validation:

```python
MODAL_RE = re.compile(r"\b(shall|must|required|obligated)\b", re.IGNORECASE)
DEADLINE_RE = re.compile(r"\b(within|no later than|\d+\s*hours?)\b", re.IGNORECASE)
ACTION_RE = re.compile(r"\b(notify|report|submit|file)\b", re.IGNORECASE)

def is_strong_obligation(text):
    """True if text has MODAL + (ACTION or DEADLINE)."""
    return MODAL_RE.search(text) and (ACTION_RE.search(text) or DEADLINE_RE.search(text))
```

---

## Evaluation Harness

### Evaluator ([eval/evaluator.py](eval/evaluator.py))

Scores results against gold standard:

```python
class Evaluator:
    def evaluate_single(self, query, response):
        result = QueryResult(...)
        
        # Check if we found the gold obligation
        for ob in result.found_obligations:
            if ob.citations[0].doc_id in valid_doc_ids:
                matched = True
                result.deadline_match = self._fuzzy_match(gold["deadline"], ob.deadline)
                result.notify_who_match = self._fuzzy_match(gold["notify_who"], ob.notify_who)
        
        result.critical_miss = not matched
        result = self._categorize_error(result, gold)
        return result
```

Fuzzy matching handles variations like "24 hours" vs "within 24 hours".

### Query Result Tracking

```python
@dataclass
class QueryResult:
    query_id: str
    critical_miss: bool
    deadline_match: Optional[bool]
    notify_who_match: Optional[bool]
    citation_valid_count: int
    false_positive_count: int
    rlm_obligations_verified: int
    rlm_obligations_dropped_by_verifier: int
    rlm_repair_attempts: int
    error_category: str  # doc_not_discovered, wrong_span, etc.
```

### Cost Tracking ([eval/experiment_config.py](eval/experiment_config.py))

```python
@dataclass
class QueryCost:
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: int = 0
    retrieval_calls: int = 0
    # RLM verification metrics
    rlm_obligations_verified: int = 0
    rlm_obligations_dropped_by_verifier: int = 0
    rlm_repair_attempts: int = 0
    rlm_fields_repaired_deadline: int = 0
    rlm_fields_repaired_notify: int = 0
    rlm_repair_rejected_deadline: int = 0
    rlm_repair_rejected_notify: int = 0
    
    def estimated_cost_usd(self):
        return (self.prompt_tokens * 2.50 + self.completion_tokens * 10.00) / 1_000_000
```

---

## Execution Flow

### run_all_experiments.py

Runs all configurations in parallel:

```python
EXPERIMENTS = ["rag", "rlm"]

def run_experiments_parallel():
    for exp in EXPERIMENTS:
        proc = subprocess.Popen(
            [python_exe, "-m", "eval.run_eval", "--experiment", exp],
            stdout=log_file,
        )
    
    # Wait for all to complete
    # Generate comparison report
```

### run_eval.py

Main entry point for single experiments:

```python
def main():
    # Load gold standard
    evaluator = Evaluator(gold_path)
    queries = evaluator.get_gold_queries()
    
    # Initialize runner based on experiment config
    if exp_config.use_rag:
        runner = RAGRunner(model, temperature)
    else:
        runner = RLMController(model)
    
    # Run each query
    for query in queries:
        response, cost = runner.research(query["activity"], [query["state"]])
        result = evaluator.evaluate_single(query, response)
        results.append(result)
    
    # Summarize and save
    summary = evaluator.summarize(results)
    evaluator.save_results(results, summary, output_path)
```

---

## File Reference

| File | Purpose |
|------|---------|
| [rag/retriever.py](rag/retriever.py) | Hybrid semantic+BM25 retrieval |
| [rag/embeddings.py](rag/embeddings.py) | Vector embedding with BGE-large |
| [rag/index.py](rag/index.py) | FAISS index loading |
| [rag/bm25_search.py](rag/bm25_search.py) | BM25 keyword search |
| [rag/schemas.py](rag/schemas.py) | Output Pydantic models |
| [rlm/rlm_controller.py](rlm/rlm_controller.py) | Controller-Driven orchestration |
| [rlm/rlm_environment.py](rlm/rlm_environment.py) | Tool implementations |
| [rlm/paragraph_index.py](rlm/paragraph_index.py) | Paragraph-level indexing |
| [eval/run_eval.py](eval/run_eval.py) | CLI entry point |
| [eval/evaluator.py](eval/evaluator.py) | Scoring logic |
| [eval/experiment_config.py](eval/experiment_config.py) | Prompts, costs, configs |
| [eval/rag_runner.py](eval/rag_runner.py) | RAG baseline runner |
| [eval/keywords.py](eval/keywords.py) | Regex patterns for filtering |
| [eval/gold_standard.json](eval/gold_standard.json) | 40 test cases |
| [run_all_experiments.py](run_all_experiments.py) | Parallel experiment runner |
