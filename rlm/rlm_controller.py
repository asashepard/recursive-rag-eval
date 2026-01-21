"""
rlm_controller.py
------------------
RLM Controller v2: Per-state extraction with hard budgets and forced termination.
Enforces:
- max_iterations = 12 globally
- max_tool_calls = 6 per state
- max_docs = 2 per state
- max_spans = 6 per state
- Extraction-only prompt (no planning)
- Obligation-ness pre-filtering
- Deduplication of exact duplicates
- State/federal separation (no federal docs in state results)

Token tracking added for fair experiment comparison (Regime A/B).
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

from openai import OpenAI

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rlm.rlm_environment import RLMEnvironment
from rag.schemas import (
    ActivityResponse,
    StateObligations,
    Obligation,
    Citation,
)

# Import QueryCost for token tracking (optional - graceful fallback if not available)
try:
    from eval.experiment_config import (
        QueryCost,
        EXTRACTION_SYSTEM_PROMPT,
        EXTRACTION_USER_TEMPLATE,
        VERIFIER_SYSTEM_PROMPT,
        VERIFIER_USER_TEMPLATE,
        REPAIR_SYSTEM_PROMPT,
        REPAIR_USER_TEMPLATE,
        verify_prompt_hash,
    )
except ImportError:
    QueryCost = None
    EXTRACTION_SYSTEM_PROMPT = None
    EXTRACTION_USER_TEMPLATE = None
    VERIFIER_SYSTEM_PROMPT = None
    VERIFIER_USER_TEMPLATE = None
    REPAIR_SYSTEM_PROMPT = None
    REPAIR_USER_TEMPLATE = None
    verify_prompt_hash = None

# Module-level hash verification (set on first init)
_PROMPT_HASH: str = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL = "gpt-4o"
MAX_ITERATIONS_GLOBAL = 12  # Hard cap for entire research
MAX_TOOL_CALLS_PER_STATE = 6
MAX_DOCS_PER_STATE = 2
MAX_SPANS_PER_STATE = 10  # Increased from 6 to improve recall for split obligations

# Secondary sources: use only for discovery, never for obligation extraction
SECONDARY_SOURCES = {
    "NARUC",  # NARUC docs are summaries/compendia, not primary authority
}

# Conservative obligation patterns (high precision)
# Matches: shall, must, required, within N, notify, report, submit
OBLIGATION_PATTERN = re.compile(
    r'\bshall\b|\bmust\b|\brequired\b|\bwithin\s+\d+|'
    r'\bnotify\b|\breport\b|\bsubmit\b',
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Prompt helpers (use canonical prompts from experiment_config)
# ---------------------------------------------------------------------------

def _get_extraction_system():
    return EXTRACTION_SYSTEM_PROMPT

def _get_extraction_user(activity: str, state: str, context: str) -> str:
    return EXTRACTION_USER_TEMPLATE.format(activity=activity, state=state, context=context)

def _get_verifier_system():
    return VERIFIER_SYSTEM_PROMPT

def _get_verifier_user(activity: str, obligation: str, quote: str, deadline: str, notify_who: str) -> str:
    return VERIFIER_USER_TEMPLATE.format(
        activity=activity,
        obligation=obligation,
        quote=quote,
        deadline=deadline or "null",
        notify_who=notify_who or "null",
    )

def _get_repair_system():
    return REPAIR_SYSTEM_PROMPT

def _get_repair_user(obligation: str, context: str) -> str:
    return REPAIR_USER_TEMPLATE.format(obligation=obligation, context=context)


class RLMController:
    """
    Per-state extraction controller with hard budgets.
    Prevents wandering by maintaining explicit progress tracking.
    
    Architecture:
    - Discovery: Find relevant documents for each state
    - Search: Find relevant spans within each document  
    - Extract: Extract obligations from each span using unified prompt
    """
    
    def __init__(
        self,
        model: str = MODEL,
        max_iterations: int = MAX_ITERATIONS_GLOBAL,
        retrieval_mode: str = "hybrid",  # "hybrid", "bm25", "semantic"
    ):
        global _PROMPT_HASH
        
        self.model = model
        self.max_iterations = max_iterations
        self.retrieval_mode = retrieval_mode
        self.client = OpenAI()
        self.env = RLMEnvironment(retrieval_mode=retrieval_mode)
        self.iteration_count = 0
        
        # Cost tracking
        self._cost = None  # QueryCost instance for tracking
        self.total_tool_calls = 0
        
        # Verify prompt hash on first init (fairness check)
        if _PROMPT_HASH is None and verify_prompt_hash is not None:
            _PROMPT_HASH = verify_prompt_hash("RLM")
    
    def research(
        self,
        activity: str,
        states: list[str] | None = None,
        utility_domain: str = "electric",
    ):
        """
        Conduct per-state research with budgets and forced progress tracking.
        
        Args:
            activity: What to research (e.g., "incident reporting")
            states: States to research (None = all available)
            utility_domain: Sector (electric, gas, etc.)
        
        Returns:
            Tuple of (ActivityResponse, QueryCost) for cost tracking
        """
        # Initialize cost tracking
        if QueryCost:
            self._cost = QueryCost()
        else:
            self._cost = None
        
        if states is None:
            states = self.env.available_states()
        
        # Add FED to all research
        target_states = list(states)
        if "FED" not in target_states:
            target_states.append("FED")
        
        print(f"\n[RLM Controller v2] Starting research: {activity}")
        print(f"  Target states: {', '.join(target_states)}")
        print(f"  Budgets: {self.max_iterations} total iterations, "
              f"{MAX_TOOL_CALLS_PER_STATE} calls/state, "
              f"{MAX_DOCS_PER_STATE} docs/state")
        
        # Track progress per state
        state_results = {}
        
        # Per-state research loop (CONTROLLER-DRIVEN)
        for state in target_states:
            if self.iteration_count >= self.max_iterations:
                print(f"\n[Global Budget] Hit max iterations ({self.max_iterations}), stopping")
                break
            
            print(f"\n[State Loop] {state}")
            
            # Research this state
            obligations = self._research_state(activity, state, utility_domain)
            
            # Determine confidence
            if obligations:
                confidence = "high" if len(obligations) >= 2 else "partial"
                explanation = None
            else:
                confidence = "low"
                explanation = f"Searched with budgets: {MAX_DOCS_PER_STATE} docs, {MAX_TOOL_CALLS_PER_STATE} tool calls"
            
            state_results[state] = StateObligations(
                obligations=obligations,
                confidence=confidence,
                not_found_explanation=explanation,
            )
            
            print(f"  Result: {len(obligations)} obligations, confidence={confidence}")
        
        # Build final response
        response = self._build_response(activity, state_results)
        
        # Update cost tracking with final counts
        if self._cost:
            self._cost.tool_calls = self.total_tool_calls
        
        return response, self._cost
    
    def _research_state(self, activity: str, state: str, utility_domain: str) -> list[Obligation]:
        """
        Research a single state within budget.
        
        Returns:
            List of obligations with citations
        """
        # Budget tracking
        tool_calls_used = 0
        docs_explored = 0
        spans_read = 0
        
        # Phase 1: Discover documents
        print(f"  [Discovery] Finding documents for {activity}...")
        doc_ids = self._discover_documents(activity, state, utility_domain)
        tool_calls_used += 1
        
        if not doc_ids:
            print(f"    No relevant documents found")
            return []
        
        print(f"    Found {len(doc_ids)} documents: {', '.join(doc_ids[:MAX_DOCS_PER_STATE])}")
        
        # Phase 2: Extract obligations from top documents
        obligations = []
        
        for doc_id in doc_ids[:MAX_DOCS_PER_STATE]:
            if tool_calls_used >= MAX_TOOL_CALLS_PER_STATE:
                print(f"    [Budget] Hit tool call limit ({MAX_TOOL_CALLS_PER_STATE})")
                break
            
            docs_explored += 1
            print(f"    [Extraction] Searching {doc_id}...")
            
            # Find relevant spans in this document
            doc_obligations = self._extract_from_document(
                doc_id,
                activity,
                state,
                span_budget=MAX_SPANS_PER_STATE - spans_read,
            )
            
            obligations.extend(doc_obligations)
            spans_read += len(doc_obligations)
            tool_calls_used += 2  # search_in_doc + get_paragraphs
            
            # Early exit if we have enough
            if len(obligations) >= 2:
                print(f"      Found {len(obligations)} obligations, moving to next state")
                break
        
        self.iteration_count += 1
        self.total_tool_calls += tool_calls_used
        
        # Deduplicate exact duplicates
        obligations = self._dedupe_obligations(obligations)
        
        return obligations
    
    def _dedupe_obligations(self, obligations: list[Obligation]) -> list[Obligation]:
        """
        Remove exact duplicate obligations based on normalized text + deadline.
        Keeps R1.1 vs R1.2 separate (different sections = different obligations).
        Only dedupes truly identical obligations.
        """
        if not obligations:
            return []
        
        seen = set()
        unique = []
        
        for ob in obligations:
            # Build dedup key: normalized obligation text + deadline + doc_id
            ob_text = (ob.obligation or "").lower().strip()
            # Remove punctuation and extra whitespace for comparison
            ob_text = re.sub(r'[^a-z0-9\s]', '', ob_text)
            ob_text = re.sub(r'\s+', ' ', ob_text)[:100]  # First 100 chars
            
            deadline = (ob.deadline or "").lower().strip()
            doc_id = ob.citations[0].doc_id if ob.citations else ""
            section = ob.citations[0].section or "" if ob.citations else ""
            
            # Include section in key to keep R1.1 vs R1.2 separate
            key = f"{ob_text}|{deadline}|{doc_id}|{section}"
            
            if key not in seen:
                seen.add(key)
                unique.append(ob)
        
        if len(obligations) != len(unique):
            print(f"      [Dedup] Removed {len(obligations) - len(unique)} duplicate(s)")
        
        return unique
    
    def _discover_documents(self, activity: str, state: str, utility_domain: str) -> list[str]:
        """
        Find relevant document IDs using retrieve_chunks.
        Returns top 2-3 unique doc_ids.
        
        State/Federal separation:
        - When state != "FED": Only return docs where doc.state == state
        - When state == "FED": Only return federal docs (FED state)
        - NARUC docs are secondary sources - used for discovery routing only,
          never for obligation extraction
        """
        # When searching for a specific state, DON'T include federal
        # When searching for "FED", only return federal docs
        is_federal_query = state == "FED"
        
        results = self.env.retrieve_chunks(
            query=activity,
            states=[state],
            utility_domain=utility_domain,
            k=8,  # Get more chunks to filter secondary sources
            include_federal=is_federal_query,  # Only include FED when querying FED
        )
        
        # Extract unique doc_ids, excluding secondary sources
        doc_ids = []
        seen = set()
        for chunk in results:
            doc_id = chunk["doc_id"]
            
            # Skip if already seen
            if doc_id in seen:
                continue
            
            # Skip secondary sources (NARUC) - they're discovery aids only
            if self._is_secondary_source(doc_id):
                continue
            
            # For state queries, verify doc is actually from that state
            if not is_federal_query:
                doc_state = self._get_doc_state(doc_id)
                if doc_state != state:
                    continue
            
            doc_ids.append(doc_id)
            seen.add(doc_id)
            
            if len(doc_ids) >= MAX_DOCS_PER_STATE:
                break
        
        return doc_ids
    
    def _is_secondary_source(self, doc_id: str) -> bool:
        """Check if doc is a secondary source (summary/compendium, not primary authority)."""
        # Check if doc_id starts with any secondary source prefix
        for prefix in SECONDARY_SOURCES:
            if doc_id.startswith(prefix):
                return True
        return False
    
    def _get_doc_state(self, doc_id: str) -> str:
        """Extract state code from doc_id (first part before _)."""
        return doc_id.split("_")[0]
    
    def _get_context_window(self, span_id: str, doc_id: str) -> str:
        """
        Get paragraph with preceding paragraph for context.
        Helps with obligations split across paragraph boundaries.
        """
        # Get all paragraphs for this doc, sorted by para_id
        doc_paras = sorted(
            [p for p in self.env.global_para_index.paragraphs if p.doc_id == doc_id],
            key=lambda x: x.para_id
        )
        
        # Find current paragraph index
        current_idx = None
        for i, p in enumerate(doc_paras):
            if p.para_id == span_id:
                current_idx = i
                break
        
        if current_idx is None:
            # Fallback: just return current paragraph
            para = self.env.get_paragraph(span_id)
            return para["text"] if para else ""
        
        # Build context: previous paragraph + current paragraph
        context_parts = []
        if current_idx > 0:
            prev_para = doc_paras[current_idx - 1]
            context_parts.append(f"[PRECEDING PARAGRAPH]\n{prev_para.text}")
        
        context_parts.append(f"[CURRENT PARAGRAPH]\n{doc_paras[current_idx].text}")
        
        return "\n\n".join(context_parts)
    
    def _expand_query(self, activity: str) -> list[str]:
        """
        Expand activity query with synonyms for better BM25 recall.
        Returns a list of query variants to try.
        """
        queries = [activity]
        
        # Common synonym expansions for regulatory search
        expansions = {
            "notification": ["report", "reporting", "notice", "notify"],
            "reporting": ["notification", "report", "notice", "notify"],
            "incident": ["event", "occurrence", "breach", "attack"],
            "breach": ["incident", "compromise", "attack", "intrusion"],
        }
        
        activity_lower = activity.lower()
        for term, synonyms in expansions.items():
            if term in activity_lower:
                for syn in synonyms:
                    variant = activity_lower.replace(term, syn)
                    if variant not in queries:
                        queries.append(variant)
        
        return queries[:3]  # Limit to 3 variants to control cost
    
    def _extract_from_document(
        self,
        doc_id: str,
        activity: str,
        state: str,
        span_budget: int,
    ) -> list[Obligation]:
        """
        Search document for relevant spans and extract obligations.
        Uses extraction-only prompt (no planning, just parsing).
        
        Pre-filters spans for obligation language to reduce LLM calls.
        """
        # Try query expansion for better BM25 recall
        query_variants = self._expand_query(activity)
        
        # Collect unique spans from all query variants
        seen_span_ids = set()
        spans = []
        for query in query_variants:
            results = self.env.search_in_doc(doc_id, query, k=min(25, span_budget * 4))
            for span in results:
                if span["span_id"] not in seen_span_ids:
                    seen_span_ids.add(span["span_id"])
                    spans.append(span)
        
        if not spans:
            print(f"      No matching spans found")
            return []
        
        # Pre-filter: only keep spans with obligation language
        # IMPORTANT: Check FULL TEXT, not just preview, to avoid missing 
        # obligations that appear later in long paragraphs
        obligation_spans = []
        for span in spans:
            full_span = self.env.get_paragraph(span["span_id"])
            if full_span and self._has_obligation_language(full_span["text"]):
                obligation_spans.append(span)
        
        if not obligation_spans:
            print(f"      Found {len(spans)} spans but none with obligation language")
            return []
        
        print(f"      Found {len(obligation_spans)} candidate spans (filtered from {len(spans)})")
        
        obligations = []
        
        for i, span in enumerate(obligation_spans[:span_budget]):
            # Retrieve full text (already fetched above, but re-fetching is cheap)
            full_span = self.env.get_paragraph(span["span_id"])
            if not full_span:
                continue
            
            # Double-check full text for obligation language
            if not self._has_obligation_language(full_span["text"]):
                print(f"        [Span {i+1}] Skipped: no obligation language in full text")
                continue
            
            # Get context window (includes preceding paragraph for split obligations)
            context_text = self._get_context_window(span["span_id"], doc_id)
            
            # Extract ALL obligations from this span with contextual window
            full_span_with_context = {**full_span, "text": context_text}
            extracted_obligations = self._extract_obligation_from_span(full_span_with_context, doc_id, activity, state)
            if extracted_obligations:
                # ---------------------------------------------------------------------------
                # VERIFICATION + REPAIR: Validate each obligation against cited evidence
                # - If activity_match fails: DROP the obligation
                # - If fields missing/not evidenced: attempt ONE repair using same paragraph
                # Use original span text for verification (not the context window)
                # ---------------------------------------------------------------------------
                for obligation in extracted_obligations:
                    verified_obligation = self._verify_and_repair(
                        obligation, full_span["text"], activity
                    )
                    
                    if verified_obligation is None:
                        # Dropped by verifier (activity_match = false)
                        ob_preview = obligation.obligation[:60] if obligation.obligation else "N/A"
                        print(f"        [Span {i+1}] DROPPED by verifier: {ob_preview}...")
                        continue
                    
                    obligations.append(verified_obligation)
                    ob_preview = verified_obligation.obligation[:60] if verified_obligation.obligation else "N/A"
                    dl = f" [deadline: {verified_obligation.deadline}]" if verified_obligation.deadline else ""
                    nw = f" [notify: {verified_obligation.notify_who[:20]}...]" if verified_obligation.notify_who else ""
                    print(f"        [Span {i+1}] Extracted: {ob_preview}...{dl}{nw}")
            else:
                print(f"        [Span {i+1}] No obligation found")
        
        return obligations
    
    def _has_obligation_language(self, text: str) -> bool:
        """
        Check if text contains obligation language.
        Uses conservative patterns: shall, must, required, within N, notify, report, submit.
        """
        return bool(OBLIGATION_PATTERN.search(text))
    
    def _verify_and_repair(
        self,
        obligation: Obligation,
        paragraph_text: str,
        activity: str,
    ) -> Optional[Obligation]:
        """
        Verify obligation against cited evidence and repair missing fields.
        
        Returns:
            - None if activity_match fails (obligation should be dropped)
            - Updated Obligation with repaired fields if successful
        """
        if self._cost:
            self._cost.rlm_obligations_verified += 1
        
        # Get the cited quote (use first citation's quote, or paragraph text as fallback)
        quote = paragraph_text[:500]  # Default to paragraph
        if obligation.citations:
            quote = obligation.citations[0].quote or paragraph_text[:500]
        
        # Step 1: Call verifier
        verification = self._call_verifier(activity, obligation, quote)
        
        if verification is None:
            # Verifier call failed, let obligation through unchanged
            return obligation
        
        # Step 2: Check activity match - DROP if false
        if not verification.get("activity_match", True):
            if self._cost:
                self._cost.rlm_obligations_dropped_by_verifier += 1
            return None
        
        # Step 3: Check if repair is needed
        needs_deadline_repair = (
            not obligation.deadline and not verification.get("deadline_present", False)
        ) or (
            obligation.deadline and not verification.get("deadline_present", False)
        )
        
        needs_notify_repair = (
            not obligation.notify_who and not verification.get("notify_present", False)
        ) or (
            obligation.notify_who and not verification.get("notify_present", False)
        )
        
        # If verifier found deadline/notify in quote, use those (they're validated)
        updated_deadline = obligation.deadline
        updated_notify = obligation.notify_who
        
        if verification.get("deadline_present") and verification.get("deadline_text"):
            # Verifier confirmed deadline is in quote - trust it
            updated_deadline = verification["deadline_text"]
        elif obligation.deadline and not verification.get("deadline_present"):
            # Extractor claimed deadline but verifier says not in quote - null it
            updated_deadline = None
            needs_deadline_repair = True
        
        if verification.get("notify_present") and verification.get("notify_text"):
            # Verifier confirmed notify_who is in quote - trust it
            updated_notify = verification["notify_text"]
        elif obligation.notify_who and not verification.get("notify_present"):
            # Extractor claimed notify_who but verifier says not in quote - null it
            updated_notify = None
            needs_notify_repair = True
        
        # Step 4: Attempt repair if needed (using ONLY the same paragraph)
        if needs_deadline_repair or needs_notify_repair:
            repaired = self._call_repair(obligation.obligation, paragraph_text)
            
            if repaired:
                if self._cost:
                    self._cost.rlm_repair_attempts += 1
                
                # Try to repair deadline
                if needs_deadline_repair and not updated_deadline:
                    rep_deadline = repaired.get("deadline")
                    rep_quote = repaired.get("deadline_quote", "")
                    if rep_deadline and rep_quote:
                        # Validate: deadline must appear in the quote
                        if self._validate_substring(rep_deadline, rep_quote, paragraph_text):
                            updated_deadline = rep_deadline
                            if self._cost:
                                self._cost.rlm_fields_repaired_deadline += 1
                            print(f"          [Repair] deadline: {rep_deadline}")
                        else:
                            if self._cost:
                                self._cost.rlm_repair_rejected_deadline += 1
                            print(f"          [Repair REJECTED] deadline not in quote")
                
                # Try to repair notify_who
                if needs_notify_repair and not updated_notify:
                    rep_notify = repaired.get("notify_who")
                    rep_quote = repaired.get("notify_who_quote", "")
                    if rep_notify and rep_quote:
                        # Validate: notify_who must appear in the quote
                        if self._validate_substring(rep_notify, rep_quote, paragraph_text):
                            updated_notify = rep_notify
                            if self._cost:
                                self._cost.rlm_fields_repaired_notify += 1
                            print(f"          [Repair] notify_who: {rep_notify}")
                        else:
                            if self._cost:
                                self._cost.rlm_repair_rejected_notify += 1
                            print(f"          [Repair REJECTED] notify_who not in quote")
        
        # Build updated obligation
        return Obligation(
            obligation=obligation.obligation,
            trigger=obligation.trigger,
            deadline=updated_deadline,
            notify_who=updated_notify,
            required_content=obligation.required_content,
            citations=obligation.citations,
            evidence_artifacts=obligation.evidence_artifacts,
            process_gates=obligation.process_gates,
        )
    
    def _validate_substring(self, value: str, quote: str, paragraph: str) -> bool:
        """
        Validate that value appears in quote or paragraph (case-insensitive).
        """
        if not value:
            return False
        value_lower = value.lower().strip()
        quote_lower = (quote or "").lower()
        para_lower = paragraph.lower()
        
        return value_lower in quote_lower or value_lower in para_lower
    
    def _call_verifier(
        self,
        activity: str,
        obligation: Obligation,
        quote: str,
    ) -> Optional[dict]:
        """
        Call LLM to verify obligation against cited evidence.
        
        Returns dict with:
            activity_match: bool
            deadline_present: bool
            deadline_text: str or None
            notify_present: bool
            notify_text: str or None
        """
        if not VERIFIER_SYSTEM_PROMPT:
            return None  # Graceful degradation if prompts not available
        
        system_prompt = _get_verifier_system()
        user_prompt = _get_verifier_user(
            activity=activity,
            obligation=obligation.obligation or "",
            quote=quote,
            deadline=obligation.deadline,
            notify_who=obligation.notify_who,
        )
        
        try:
            api_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0,  # Deterministic verification
            }
            
            response = self.client.chat.completions.create(**api_params)
            
            # Track token usage
            if self._cost and response.usage:
                self._cost.add_llm_call(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    call_type="verification",
                )
            
            content = response.choices[0].message.content
            return json.loads(content)
        
        except (json.JSONDecodeError, ValueError, KeyError):
            return None
    
    def _call_repair(self, obligation_text: str, paragraph_text: str) -> Optional[dict]:
        """
        Call LLM to extract missing deadline/notify_who from paragraph.
        
        Returns dict with:
            deadline: str or None
            notify_who: str or None
            deadline_quote: str or None
            notify_who_quote: str or None
        """
        if not REPAIR_SYSTEM_PROMPT:
            return None  # Graceful degradation if prompts not available
        
        system_prompt = _get_repair_system()
        user_prompt = _get_repair_user(obligation_text, paragraph_text)
        
        try:
            api_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0,  # Deterministic repair
            }
            
            response = self.client.chat.completions.create(**api_params)
            
            # Track token usage
            if self._cost and response.usage:
                self._cost.add_llm_call(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    call_type="repair",
                )
            
            content = response.choices[0].message.content
            return json.loads(content)
        
        except (json.JSONDecodeError, ValueError, KeyError):
            return None
    
    def _extract_obligation_from_span(
        self,
        span: dict,
        doc_id: str,
        activity: str,
        state: str,
    ) -> list[Obligation]:
        """
        Extract ALL obligations from a single span using unified prompts.
        Uses same system prompt + user template as RAG for fair comparison.
        Returns a list of obligations (may be empty or contain multiple).
        """
        span_text = span["text"]
        
        # Build prompt using SHARED templates (same as RAG)
        system_prompt = _get_extraction_system()
        user_prompt = _get_extraction_user(activity, state, span_text)
        
        # Call LLM for extraction
        try:
            api_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
            }
            # Only set temperature if model supports it
            if "nano" not in self.model.lower():
                api_params["temperature"] = 0.1
            
            response = self.client.chat.completions.create(**api_params)
            
            # Track token usage
            if self._cost and response.usage:
                self._cost.add_llm_call(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    call_type="extraction",
                )
            
            # Parse response - unified schema: {"obligations": [...]}
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Handle unified schema: obligations array
            obligations_data = result.get("obligations", [])
            if not obligations_data:
                return []
            
            # Extract ALL obligations from the paragraph (not just the first)
            all_obligations = []
            for obl_data in obligations_data:
                if not obl_data.get("obligation"):
                    continue
                
                # Build citation - ALWAYS use the known doc_id we passed in
                # (LLM doesn't know the real doc_id, so ignore whatever it returns)
                citations = []
                for cite_data in obl_data.get("citations", []):
                    citations.append(Citation(
                        doc_id=doc_id,  # Always use known doc_id
                        page=span.get("page"),
                        section=span.get("section_heading"),
                        quote=cite_data.get("quote", span_text[:200]),
                    ))
                
                # Fallback citation if LLM didn't provide one
                if not citations:
                    citations = [Citation(
                        doc_id=doc_id,
                        page=span.get("page"),
                        section=span.get("section_heading"),
                        quote=span_text[:200],
                    )]
                
                # Build obligation with unified fields
                obligation = Obligation(
                    obligation=obl_data.get("obligation"),
                    trigger=obl_data.get("trigger"),
                    deadline=obl_data.get("deadline"),
                    notify_who=obl_data.get("notify_who"),
                    required_content=obl_data.get("required_content"),
                    citations=citations,
                    evidence_artifacts=[],
                    process_gates=[],
                )
                all_obligations.append(obligation)
            
            return all_obligations
        
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Extraction failed; silently skip this span
            return []
    
    def _build_response(
        self,
        activity: str,
        state_results: dict,
    ) -> ActivityResponse:
        """Build final ActivityResponse from state results."""
        return ActivityResponse(
            activity=activity,
            states=state_results,
            diffs=[],  # TODO: implement diff generation
            federal_baseline=state_results.get("FED"),
        )


def main():
    """Example usage."""
    print("Initializing RLM Controller...")
    controller = RLMController(model="gpt-4o", max_iterations=MAX_ITERATIONS_GLOBAL)
    
    print("\nStarting research...")
    result, cost = controller.research(
        activity="cybersecurity incident reporting",
        states=["TX", "MD"],
    )
    
    print("\n" + "=" * 60)
    print(f"RESEARCH COMPLETE: {controller.iteration_count} iterations, "
          f"{controller.total_tool_calls} tool calls")
    if cost:
        print(f"COST: {cost.llm_calls} LLM calls, {cost.total_tokens} tokens, "
              f"${cost.to_dict()['estimated_cost_usd']:.4f} est.")
    print("=" * 60)
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
