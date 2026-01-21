# Experiment Guide

**Authoritative reference for running and scoring the RAG vs. Controller-Driven comparison experiment.**

This document covers metrics, scoring logic, and how to interpret results. For implementation details, see [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md).

---

## Running the Experiment

### Full Suite (Recommended)

```bash
python run_all_experiments.py
```

This runs both configurations in parallel and generates a comparison report in `eval/results/`.

### Individual Experiments

```bash
# RAG baseline
python -m eval.run_eval --experiment rag

# Controller-Driven with iterative search
python -m eval.run_eval --experiment rlm
```

### Code Layout Note (Why `eval/rag_runner.py` exists)

- The experiment harness lives under `eval/` and is intended to be run as a module (e.g., `python -m eval.run_eval`).
- The RAG baseline implementation is therefore in [eval/rag_runner.py](eval/rag_runner.py) so it can be imported by the harness and share the same prompts, schema, and cost tracking.
- The top-level scripts like [run_rlm.py](run_rlm.py) (and [run_rag.py](run_rag.py)) are ad-hoc utilities for manual debugging and one-off queries; they are not part of the scored harness.

### Dry Run (Preview)

```bash
python -m eval.run_eval --experiment rag --dry-run
```

Shows which queries would run without executing them.

---

## Experiment Configurations

| Config | Strategy | Evidence Consumption |
|--------|----------|----------------------|
| `rag` | Single-pass RAG | Retrieve top-k chunks → 1 LLM call |
| `rlm` | Controller-Driven | Discover → search → extract → verify → repair |

**No Pre-LLM Gating:** Neither strategy uses keyword-based abstain gates. When retrieval returns chunks, the LLM is always called. Negative test behavior relies solely on prompt instructions ("return empty array if no obligations found").

**Controller-Driven Verification Loop:** After extracting an obligation, Controller-Driven runs a verification step:
1. **Activity match** - Does the obligation actually relate to the query activity?
2. **Deadline present** - Is the deadline field supported by the cited text?
3. **Notify present** - Is the notify_who field supported by the cited text?

If activity_match is false, the obligation is dropped. If deadline/notify are missing but should exist, a **repair step** attempts to extract them from the same paragraph, accepting the repair only if the extracted substring appears verbatim in the text.

---

## Federal Baseline Policy

State-specific queries (e.g., "incident reporting in NJ") must return **both state AND federal obligations**.

**Rationale:** Federal regulations (NERC CIP, DOE) apply to all utilities regardless of state. A NJ utility must comply with both NJ-specific rules and federal requirements.

**Scoring rules:**
- Federal obligations from valid FED documents count as correct (not penalized)
- Both state and federal obligations must be returned for complete answers
- The extraction prompt explicitly instructs both strategies to return all applicable obligations

**Implementation:** Encoded in the extraction prompt ([experiment_config.py](eval/experiment_config.py)), gold standard metadata ([gold_standard.json](eval/gold_standard.json)), and evaluator scoring ([evaluator.py](eval/evaluator.py)).

---

## How Scoring Works

1. **Load gold standard:** 40 queries with known correct answers (28 positive, 12 negative)
2. **Run strategy:** Execute RAG or Controller-Driven on each query
3. **Match obligations:** For positive cases, check if returned obligation matches gold (by doc_id + fuzzy text match)
4. **Score fields:** If obligation found, check deadline and notify_who accuracy
5. **Categorize errors:** If missed, categorize why (doc not discovered, wrong span, etc.)
6. **Aggregate:** Compute rates across all queries

Scoring logic: [eval/evaluator.py](eval/evaluator.py)

---

## Why the Comparison Is Fair

The experiment controls for everything except the evidence consumption strategy:

### Identical Across All Configs

| Factor | How Controlled | Verification |
|--------|----------------|--------------|
| **LLM Model** | Defaults to `gpt-4o`; override with `--model` flag | Same model used across all configs in a run |
| **Temperature** | Hardcoded `0` (minimizes LLM sampling variance) | Config file |
| **Extraction Prompt** | Both use `EXTRACTION_SYSTEM_PROMPT` and `EXTRACTION_USER_TEMPLATE` | Hash verified at runtime |
| **Output Schema** | Both produce `{"obligations": [...]}` per [schemas.py](rag/schemas.py) | Same Pydantic models |
| **Corpus** | Same 17 documents, same chunking, same indexes | Single corpus directory |
| **Scoring Logic** | Same `Evaluator` class scores all results | [evaluator.py](eval/evaluator.py) |

### Prompt Hash Verification

At initialization, both RAGRunner and RLMController call `verify_prompt_hash()` in [experiment_config.py](eval/experiment_config.py). This computes a SHA-256 hash of the system prompt and schema, ensuring both systems use identical prompts:

```
[RAG] Extraction prompt hash: a1b2c3d4e5f6
[RLM] Extraction prompt hash: a1b2c3d4e5f6
```

If the hashes differ, the experiment fails.

### Budget Caps

Controller-Driven has hard limits to prevent "winning by spending more":

| Budget | Value | Enforced In |
|--------|-------|-------------|
| Max docs per state | 2 | [rlm_controller.py](rlm/rlm_controller.py) |
| Max spans per state | 10 | [rlm_controller.py](rlm/rlm_controller.py) |
| Max tool calls per state | 6 | [rlm_controller.py](rlm/rlm_controller.py) |
| Max global iterations | 12 | [rlm_controller.py](rlm/rlm_controller.py) |

### Controller-Driven Retrieval Enhancements

Controller-Driven uses several techniques to achieve parity with RAG's semantic capabilities:

| Technique | Purpose |
|-----------|--------|
| **Query Expansion** | Adds synonyms (notification ↔ report, incident ↔ breach) to catch term mismatches |
| **Contextual Windowing** | Includes preceding paragraph to catch obligations split across boundaries |
| **Full-text Filtering** | Checks entire paragraph for obligation language, not just the preview |
| **Multi-obligation Extraction** | Extracts ALL obligations from a paragraph, not just the first match |
| **Increased k** | Retrieves top-25 candidates from within-doc BM25 search |

### Why These Enhancements Are Fair

These enhancements achieve **parity** with RAG's inherent advantages, not an unfair edge:

| Enhancement | Why It's Fair |
|-------------|---------------|
| **Query Expansion** | RAG uses hybrid semantic+BM25 retrieval. Semantic embeddings *already* handle synonyms implicitly ("notification" ≈ "report" in embedding space). Controller-Driven's BM25-only within-doc search needs explicit expansion to match this capability. |
| **Contextual Windowing** | RAG uses 900-token overlapping chunks, naturally providing cross-boundary context. Controller-Driven's paragraph-level granularity needs explicit windowing to achieve similar context. |
| **Multi-obligation Extraction** | Both strategies use identical extraction prompts that request ALL obligations. This fix ensures Controller-Driven's paragraph-by-paragraph approach doesn't artificially limit what RAG's single-pass naturally captures. |
| **Increased k** | RAG retrieves top-12 from each index (24 candidates), then reranks to 6. Controller-Driven's k=25 is comparable candidate volume before filtering. |

The core variable under test—**evidence consumption strategy**—remains isolated.

---

## Gold Standard

The test set contains 40 queries in [gold_standard.json](eval/gold_standard.json):

- **28 positive cases:** An obligation exists; we verify the system finds it
- **12 negative cases:** No obligation exists; we verify the system returns nothing

Each positive case specifies:

```json
{
  "id": "NJ_incident_reporting",
  "activity": "incident reporting",
  "state": "NJ",
  "must_catch": {
    "obligation": "Report cyber incidents within 6 hours",
    "deadline": "6 hours after detection",
    "notify_who": "Reliability and Security Division Staff via NJCCIC",
    "cite": {
      "doc_id": "NJ_NJ_3-18-16-6A",
      "corpus_text": "Pursuant to N.JAC. 14:3-6.7, reports shall be made within 6 hours..."
    }
  }
}
```

The `corpus_text` field contains the exact quote from the document, allowing verification that the obligation actually exists in the corpus.

---

## Metrics

### Primary Metrics

| Metric | Definition | Good = |
|--------|------------|--------|
| **Critical Miss Rate** | Fraction of positive cases where the system failed to find the obligation | Lower |
| **Negative Test Accuracy** | Fraction of negative cases where the system correctly returned no obligations | Higher |

### Secondary Metrics

| Metric | Definition | Notes |
|--------|------------|-------|
| **Deadline Accuracy** | Fraction of found obligations with correct deadline | Only counted when obligation was found |
| **Notify Who Accuracy** | Fraction of found obligations with correct recipient | Only counted when obligation was found |
| **Citation Validity** | Fraction of citations pointing to real documents | Should be 100% |
| **False Obligation Rate** | Fraction of returned obligations not in gold standard | Lower is better |

### Cost Metrics

| Metric | Definition |
|--------|------------|
| **LLM Calls** | Total API calls to OpenAI |
| **Total Tokens** | Prompt + completion tokens |
| **Estimated Cost** | USD at current GPT-4o pricing |

### Controller-Driven Verification Metrics

| Metric | Definition |
|--------|------------|
| **RLM Verified** | Obligations that passed verification |
| **RLM Dropped** | Obligations dropped because activity didn't match |
| **Repair Attempts** | Times repair was called for missing fields |
| **Repaired Deadline** | Deadlines successfully extracted by repair |
| **Repaired Notify** | Notify_who successfully extracted by repair |
| **Rejected Deadline** | Deadline repairs rejected (substring not in text) |
| **Rejected Notify** | Notify repairs rejected (substring not in text) |

---

## Interpreting Results

The comparison report shows all metrics side-by-side:

```
Metric                         rag       rlm
------------------------------------------------------
Critical Miss Rate            21%       18%
Negative Test Accuracy        58%       75%
Deadline Accuracy             32%       45%
RLM Verified                   --        42
RLM Dropped by Verifier        --         3
LLM Calls                      34       166
Estimated Cost (USD)        $0.63     $1.06
```

### What to Look For

**Controller-Driven wins if:**
- Critical miss rate is lower (finds more obligations)
- At acceptable cost increase
- Verification drops few legitimate obligations

**Controller-Driven worse than RAG if:**
- Same or higher critical miss rate
- At higher cost

### Error Categories

When a system misses an obligation, the evaluator categorizes why:

| Category | Meaning |
|----------|---------|
| `doc_not_discovered` | The correct document was never retrieved |
| `wrong_span` | Found the right document but wrong section |
| `extractor_missed_field` | Found the obligation but missed deadline/notify_who |
| `false_positive` | Returned an obligation that doesn't exist |

The comparison report breaks down errors by category, helping diagnose where each system fails.

---

## Output Files

Results are saved to `eval/results/`:

```
eval_rag_20260120_143022.json       # Detailed RAG results
eval_rlm_20260120_143156.json       # Detailed RLM results
comparison_report_20260120_143200.txt  # Side-by-side comparison
log_rag_20260120_143022.txt         # Console output for debugging
```

The JSON files contain per-query details including citations, field matches, and error categories.

---

## Common Issues

### "OPENAI_API_KEY not set"

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."

# Or create a .env file in the project root
echo "OPENAI_API_KEY=sk-..." > .env
```

### Running with a different model (experimental)

```bash
python run_all_experiments.py --model gpt-4o-mini
```

**Important:** All configs in a single run use the same model, ensuring fair comparison within that run. However, results from different models are not directly comparable—weaker models may struggle with the iterative Controller-Driven approach while performing adequately on single-shot RAG.

### Filtering to specific states

```bash
python -m eval.run_eval --experiment rlm --states NJ MD TX
```
