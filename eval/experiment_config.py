"""
experiment_config.py
--------------------
Unified settings and prompts for fair experiment comparison.

FAIRNESS DESIGN:
- RAG and RLM use IDENTICAL system prompt and user template
- Same rules: "do not speculate", citation requirements, null handling
- Same output schema: {"obligations": [...]} with citations
- Only difference: context size (top-k chunks vs paragraph)

Locked Settings (never vary across experiments):
  - Model: GPT-4o
  - Temperature: 0 (deterministic)  
  - System prompt: Same rules for all extraction
  - User template: Same structure, variable context
  - Output schema: {"obligations": [...]} with citations
  - Scoring: Same evaluator logic
"""

from dataclasses import dataclass, field
import hashlib
import json


# ---------------------------------------------------------------------------
# Locked Settings (NEVER vary across experiments)
# ---------------------------------------------------------------------------
MODEL = "gpt-4o"
TEMPERATURE = 0  # Deterministic for reproducibility
MAX_OUTPUT_TOKENS = 4096

# ---------------------------------------------------------------------------
# UNIFIED PROMPTS - Used by BOTH RAG and RLM for fair comparison
# ---------------------------------------------------------------------------

# Extraction schema (used for hash verification)
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "obligations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "obligation": {"type": "string"},
                    "trigger": {"type": ["string", "null"]},
                    "deadline": {"type": ["string", "null"]},
                    "notify_who": {"type": ["string", "null"]},
                    "required_content": {"type": ["string", "null"]},
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "doc_id": {"type": "string"},
                                "quote": {"type": "string"}
                            },
                            "required": ["doc_id", "quote"]
                        }
                    }
                },
                "required": ["obligation", "citations"]
            }
        }
    },
    "required": ["obligations"]
}

# System prompt: Same rules for all extraction calls
# GOLDEN RULES: Minimal set for verifiable compliance extraction.
# Each rule is enforceable via schema validation, regex patterns, or quote checks.
EXTRACTION_SYSTEM_PROMPT = """You are a regulatory compliance expert. Extract structured obligations from the provided regulatory text.

RULES:
1. Only extract obligations EXPLICITLY stated in the provided text. Do NOT infer, generalize, or speculate. Use null for any field not explicitly stated.
2. If the text contains no compliance obligations, return an empty array: {"obligations": []}.
3. Every obligation MUST include a citation with doc_id and an exact quote (max 200 chars) from the source text.
4. Every non-null field (deadline, notify_who, required_content) must be supported by a citation quote containing that value.
   - For deadline: the quote must contain the timeframe (e.g., "within 24 hours", "1 business day").
   - For notify_who: the quote must contain the recipient name/role.
5. Do not use information from outside the provided text.

Output ONLY valid JSON matching the schema."""

# User template: Same structure for RAG (top-k chunks) and RLM (paragraph)
EXTRACTION_USER_TEMPLATE = """QUERY: What are the {activity} requirements for {state}?

TASK DEFINITION:
- For state-specific queries (NJ, MD, etc.), return BOTH state-specific AND federal baseline obligations.
- Federal requirements (NERC CIP, DOE) apply to all regulated utilities regardless of state.
- A complete answer includes all applicable obligations from both state and federal sources.

REGULATORY TEXT:
{context}

Extract all compliance obligations from the text above.

For each obligation, provide:
- obligation: What must be done
- trigger: Event/condition that triggers the requirement
- deadline: Timeframe (e.g., "24 hours", "1 business day", or null)
- notify_who: Who must be notified (or null)
- required_content: What information must be included (or null)
- citations: Array of {{"doc_id": "...", "quote": "exact text..."}}

OUTPUT valid JSON:
{{
  "obligations": [
    {{
      "obligation": "description",
      "trigger": "triggering event or null",
      "deadline": "timeframe or null",
      "notify_who": "recipient or null",
      "required_content": "content requirements or null",
      "citations": [{{"doc_id": "...", "quote": "..."}}]
    }}
  ]
}}

Output ONLY valid JSON, no explanation."""


# ---------------------------------------------------------------------------
# VERIFICATION PROMPT - Validates extracted obligations against cited evidence
# ---------------------------------------------------------------------------
VERIFIER_SYSTEM_PROMPT = """You verify whether a compliance obligation is supported by the cited evidence.

RULES:
1. You ONLY check what is EXPLICITLY present in the provided quote/text.
2. Do not infer or assume. If the text does not contain the exact information, answer false.
3. For deadline: the quote must contain a specific timeframe (e.g., "24 hours", "1 business day", "immediately").
4. For notify_who: the quote must contain a specific recipient name, role, or organization.
5. For activity_match: does the quoted text describe a requirement related to the stated activity?

Output ONLY valid JSON."""

VERIFIER_USER_TEMPLATE = """ACTIVITY: {activity}

EXTRACTED OBLIGATION: {obligation}

CITED QUOTE:
{quote}

CURRENT FIELDS:
- deadline: {deadline}
- notify_who: {notify_who}

Verify the following by examining ONLY the cited quote:

1. activity_match: Does the quote describe a requirement related to "{activity}"? (true/false)
2. deadline_present: Does the quote contain a specific timeframe/deadline? (true/false)
3. deadline_text: If deadline_present is true, what is the exact deadline substring from the quote? (or null)
4. notify_present: Does the quote contain a specific recipient to notify? (true/false)
5. notify_text: If notify_present is true, what is the exact recipient substring from the quote? (or null)

OUTPUT valid JSON:
{{
  "activity_match": true/false,
  "deadline_present": true/false,
  "deadline_text": "exact substring or null",
  "notify_present": true/false,
  "notify_text": "exact substring or null"
}}

Output ONLY valid JSON, no explanation."""

# ---------------------------------------------------------------------------
# REPAIR PROMPT - Extracts missing fields from the same paragraph
# ---------------------------------------------------------------------------
REPAIR_SYSTEM_PROMPT = """You extract ONLY the deadline and notify_who fields from regulatory text.

RULES:
1. Only extract values EXPLICITLY stated in the provided text.
2. For deadline: must be a specific timeframe (e.g., "24 hours", "1 business day", "immediately").
3. For notify_who: must be a specific entity, role, or organization name.
4. If a field is not explicitly stated, use null.
5. Include an exact quote (max 150 chars) containing the extracted value.

Output ONLY valid JSON."""

REPAIR_USER_TEMPLATE = """OBLIGATION: {obligation}

REGULATORY TEXT:
{context}

Extract ONLY the following fields if they are EXPLICITLY stated in the text:

OUTPUT valid JSON:
{{
  "deadline": "timeframe or null",
  "notify_who": "recipient or null",
  "deadline_quote": "exact quote containing deadline or null",
  "notify_who_quote": "exact quote containing recipient or null"
}}

Output ONLY valid JSON, no explanation."""


# ---------------------------------------------------------------------------
# Prompt Hash Verification (for fairness auditing)
# ---------------------------------------------------------------------------
def get_extraction_prompt_hash() -> str:
    """
    Generate a hash of the extraction prompt + schema for fairness verification.
    RAG and RLM should produce the same hash to confirm identical prompts.
    """
    content = EXTRACTION_SYSTEM_PROMPT + json.dumps(EXTRACTION_SCHEMA, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def verify_prompt_hash(component: str, expected_hash: str = None) -> str:
    """
    Log and optionally verify prompt hash for a component.
    Call this at runner initialization to confirm fair comparison.
    
    Args:
        component: Name of component (e.g., "RAG", "RLM")
        expected_hash: If provided, assert hash matches
        
    Returns:
        The computed hash
    """
    computed = get_extraction_prompt_hash()
    print(f"  [{component}] Extraction prompt hash: {computed}")
    
    if expected_hash and computed != expected_hash:
        raise ValueError(
            f"Prompt hash mismatch for {component}! "
            f"Expected {expected_hash}, got {computed}. "
            f"This indicates prompts are not unified across systems."
        )
    
    return computed


# ---------------------------------------------------------------------------
# Budget Caps (prevents RLM from "cheating" with unlimited calls)
# ---------------------------------------------------------------------------
@dataclass
class BudgetCaps:
    """Hard budget limits for fair comparison."""
    max_docs_per_state: int = 2
    max_spans_per_state: int = 6
    max_tool_calls_per_state: int = 6
    max_iterations_global: int = 12
    max_total_llm_calls: int = 40  # Total LLM API calls per query

DEFAULT_BUDGET = BudgetCaps()


# ---------------------------------------------------------------------------
# Cost Tracking
# ---------------------------------------------------------------------------
@dataclass
class QueryCost:
    """Track cost for a single query execution."""
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tool_calls: int = 0
    retrieval_calls: int = 0

    # RAG-specific metrics
    rag_abstained_count: int = 0
    
    # RLM verification/repair metrics
    rlm_obligations_verified: int = 0  # Obligations that went through verifier
    rlm_obligations_dropped_by_verifier: int = 0  # Dropped due to activity_match=false
    rlm_repair_attempts: int = 0  # Repair calls made
    rlm_fields_repaired_deadline: int = 0  # Deadlines successfully repaired
    rlm_fields_repaired_notify: int = 0  # Notify_who successfully repaired
    rlm_repair_rejected_deadline: int = 0  # Deadline repairs rejected (not in quote)
    rlm_repair_rejected_notify: int = 0  # Notify repairs rejected (not in quote)
    
    def add_llm_call(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        call_type: str = "extraction",
    ) -> None:
        """Record an LLM API call."""
        self.llm_calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
    
    def add_tool_call(self) -> None:
        """Record a tool call (retrieval)."""
        self.tool_calls += 1
    
    def add_retrieval(self) -> None:
        """Record a retrieval operation."""
        self.retrieval_calls += 1
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "tool_calls": self.tool_calls,
            "retrieval_calls": self.retrieval_calls,
            "rag_abstained_count": self.rag_abstained_count,
            # RLM verification/repair metrics
            "rlm_obligations_verified": self.rlm_obligations_verified,
            "rlm_obligations_dropped_by_verifier": self.rlm_obligations_dropped_by_verifier,
            "rlm_repair_attempts": self.rlm_repair_attempts,
            "rlm_fields_repaired_deadline": self.rlm_fields_repaired_deadline,
            "rlm_fields_repaired_notify": self.rlm_fields_repaired_notify,
            "rlm_repair_rejected_deadline": self.rlm_repair_rejected_deadline,
            "rlm_repair_rejected_notify": self.rlm_repair_rejected_notify,
            # Estimated cost at $2.50/$10 per 1M tokens (GPT-4o)
            "estimated_cost_usd": (
                self.prompt_tokens * 2.50 / 1_000_000 +
                self.completion_tokens * 10.00 / 1_000_000
            ),
        }


# ---------------------------------------------------------------------------
# Experiment Configuration
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    """Configuration for an ablation experiment."""
    name: str
    description: str
    
    # Core switch: RAG vs RLM
    use_rag: bool = False  # True = single-pass RAG, False = RLM controller
    
    # RLM-specific settings (ignored if use_rag=True)
    retrieval_mode: str = "hybrid"
    
    # Budget (Regime B)
    budget: BudgetCaps = field(default_factory=BudgetCaps)
    
    def __post_init__(self):
        """Ensure budget is always a BudgetCaps instance."""
        if not isinstance(self.budget, BudgetCaps):
            self.budget = BudgetCaps()


# ---------------------------------------------------------------------------
# Experiment Configurations
# ---------------------------------------------------------------------------
# Two configs for fair comparison:
#
#   rag:  "What does a strong single-pass RAG system do?"
#   rlm:  "Does multi-doc iterative search improve over single-pass RAG?"

EXPERIMENT_CONFIGS = {
    "rag": ExperimentConfig(
        name="rag",
        description="Single-pass RAG baseline (retrieve top-k, one LLM call)",
        use_rag=True,
    ),
    "rlm": ExperimentConfig(
        name="rlm",
        description="RLM controller with iterative multi-doc search",
        use_rag=False,
    ),
}


def get_experiment_config(name: str) -> ExperimentConfig:
    """Get experiment config by name."""
    if name not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment: {name}. Valid: {list(EXPERIMENT_CONFIGS.keys())}")
    return EXPERIMENT_CONFIGS[name]
