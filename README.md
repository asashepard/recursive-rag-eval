# RLM vs RAG: A Controlled Experiment

**Does breaking up document retrieval into smaller, iterative steps improve extraction accuracy—or just cost more?**

This repository contains a controlled experiment comparing two approaches to extracting structured compliance obligations from regulatory documents.

The RLM (Recursive Language Model) approach is inspired by ["Recursive Language Models"](https://arxiv.org/abs/2512.24601) (Zhang, Kraska & Khattab, 2025), which treats long prompts as part of an external environment that the LLM can programmatically examine and decompose.

### How This Differs From the Paper

The paper's RLM has the LLM write Python code in a REPL to slice context, choose chunking strategies, and recursively call itself on snippets. This implementation uses a **controller-driven** variant:

| Aspect | Paper's RLM | This Implementation |
|--------|-------------|---------------------|
| **Control flow** | LLM writes code to decide next action | Fixed loop: discover → search → extract |
| **Context access** | LLM slices strings (`context[1000:2000]`) | Controller calls `env.get_paragraph(id)` |
| **Recursion** | LLM calls `llm_query()` on self-selected snippets | Controller calls LLM on each paragraph |
| **Termination** | LLM outputs `FINAL()` when satisfied | Controller stops after budget exhausted |

**Why controller-driven?** For a controlled experiment, this design is actually preferable:

- **Reproducibility:** Same steps every run (LLM-driven would vary)
- **Fair comparison:** Isolates the retrieval strategy variable without confounding LLM coding ability
- **Budget enforcement:** Deterministic cost control (paper shows 10× variance between 25th–95th percentile)

The core hypothesis—*"processing small spans iteratively beats stuffing everything into one prompt"*—is preserved. The mechanism differs, but the variable being tested is the same: **evidence consumption strategy**.

---

## The Problem

Imagine you're a utility company that needs to know: *"If we have a cybersecurity incident in New Jersey, who do we notify, and how quickly?"*

The answer is buried somewhere in regulatory documents—PDFs from state public utility commissions, NERC standards, DOE forms. A typical RAG system retrieves relevant chunks and asks an LLM to extract the answer. But regulatory text is dense, cross-referenced, and easy to misread.

**This experiment asks:** Can we do better by searching more carefully?

---

## Two Approaches

### RAG (Retrieval-Augmented Generation)
The standard approach. Retrieve the top-10 most relevant document chunks, concatenate them, and ask the LLM to extract obligations in one shot.

- **Pros:** Fast, cheap, simple
- **Cons:** May miss details buried in long context; no recovery if first retrieval is wrong

### RLM (Recursive Language Model)
An iterative approach. Search for relevant documents, then search within each document for specific paragraphs, extract from each paragraph separately. After extraction, a **verification step** checks that each obligation is actually supported by the cited evidence, and a **repair step** can fill missing fields (deadline, notify_who) from the same paragraph.

**Key techniques:**
- **Query Expansion:** Adds synonym variants (notification ↔ report, incident ↔ breach) to improve BM25 recall
- **Contextual Windowing:** Includes the preceding paragraph to catch obligations split across boundaries
- **Multi-obligation Extraction:** Extracts ALL obligations from a paragraph, not just the first match

- **Pros:** Focused attention on smaller text spans; verification prevents hallucinated obligations; repair fills missing fields; query expansion finds more relevant paragraphs
- **Cons:** Slower, more expensive (multiple LLM calls)

---

## What We Measure

Given a query like "incident reporting requirements in NJ", both systems must return:

```json
{
  "obligation": "Report cyber incidents to NJCCIC",
  "deadline": "6 hours",
  "notify_who": "Reliability and Security Division",
  "citations": [{"doc_id": "NJ_NJ_3-18-16-6A", "quote": "..."}]
}
```

We have 40 test cases with known correct answers (28 positive cases where an obligation exists, 12 negative cases where no obligation should be found). We measure:

| Metric | What It Tells You |
|--------|-------------------|
| **Critical Miss Rate** | How often the system fails to find an obligation that exists |
| **Negative Test Accuracy** | How often the system correctly returns nothing when nothing exists |
| **Deadline Accuracy** | How often the extracted deadline matches the gold standard |
| **Notify Who Accuracy** | How often the extracted recipient matches the gold standard |
| **Cost** | Tokens used, LLM calls, estimated USD |

---

## Why the Comparison Is Fair

Both systems use:
- The same LLM (GPT-4o)
- The same extraction prompt (verified by hash)
- The same output schema
- The same temperature (0, for deterministic output)
- The same corpus (17 regulatory documents)
- The same scoring logic

The **only difference** is how they consume evidence: RAG sees many chunks at once; RLM sees smaller spans iteratively.

### Federal Baseline Policy

For state-specific queries (NJ, MD, etc.), both systems return **both state-specific AND federal baseline obligations**. Federal requirements (NERC CIP, DOE) apply to all regulated utilities regardless of state. This means:

- A query for "incident reporting in NJ" will return NJ-specific rules AND federal NERC/DOE requirements
- Federal obligations returned for state queries are **not** counted as false positives
- This matches real-world compliance needs: a NJ utility must comply with both state and federal rules

For details on fairness controls, see [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md).

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Re-download the source documents and rebuild the corpus end-to-end
python build_corpus.py --download

# Run the full experiment suite (2 configurations: rag, rlm)
python run_all_experiments.py

# Results appear in eval/results/comparison_report_*.txt
```

---

## Repository Structure

```
rlm-utilities/
├── corpus/normalized/     # 17 processed regulatory documents
├── rag/                   # RAG retrieval components
├── rlm/                   # RLM controller and tools
├── eval/                  # Evaluation harness and gold standard
├── run_all_experiments.py # Runs all configs and generates comparison
└── requirements.txt
```

For a detailed walkthrough of every component, see [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md).

---

## Utility Scripts

These scripts are not part of the evaluation harness, but are useful during development.

- [run_rlm.py](run_rlm.py): Run a single ad-hoc RLM query and save the raw JSON output (good for debugging without running the full eval suite).
- [run_rag.py](run_rag.py): Run a single ad-hoc RAG query using the same prompt/schema as the experiment baseline.
- [build_corpus.py](build_corpus.py): Rebuild the `corpus/` directory from `downloads/_download-log.csv` by extracting text/metadata from downloaded source files.
- [urls.txt](urls.txt): Canonical list of public source URLs used to assemble the corpus.
- [download_sources.py](download_sources.py): Download `urls.txt` into `downloads/` and regenerate `downloads/_download-log.csv`.

---

## Interpreting Results

The comparison report shows metrics side-by-side:

```
Metric                         rag       rlm
------------------------------------------------------
Critical Miss Rate            21%       18%
Negative Test Accuracy        58%       75%
Deadline Accuracy             32%       45%
RLM Verified                   --        42
RLM Dropped by Verifier        --         3
Cost (USD)                  $0.63     $1.06
```

What this tells you:
- If RLM has a **lower** critical miss rate at reasonable cost, the iterative approach is worth it
- If RLM has **higher** negative test accuracy, it hallucinates less
- **RLM Verified/Dropped** shows how many obligations passed or failed the verification step
- If the costs are similar, the winner is whichever has better accuracy

---

## Who This Is For

- **Researchers** exploring alternatives to single-shot RAG
- **ML Engineers** evaluating multi-step extraction pipelines
- **Compliance teams** considering automation for regulatory obligation tracking

---

## License

MIT — see [LICENSE](LICENSE)
