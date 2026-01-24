"""
rag_runner.py
-------------
Fair RAG baseline runner for experiment comparison.

Key differences from rag_pipeline.py:
- Uses SAME extraction prompt as RLM (not the long system prompt)
- Tracks token usage for cost comparison
- Returns ActivityResponse compatible with evaluator
- Single-pass: retrieve top-k chunks, one LLM call

This ensures fair comparison: RAG vs RLM differ only in architecture,
not in prompts or schema.
"""

import json
from typing import Optional

from openai import OpenAI

from rag.retriever import Retriever
from rag.schemas import (
    ActivityResponse,
    StateObligations,
    Obligation,
    Citation,
)
from eval.experiment_config import (
    MODEL,
    TEMPERATURE,
    MAX_OUTPUT_TOKENS,
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_TEMPLATE,
    QueryCost,
    verify_prompt_hash,
)
# Keyword gating removed - LLM always called when retrieval returns chunks.
# NEG behavior relies on prompt instruction: "return empty array if no obligations".


# ---------------------------------------------------------------------------
# RAG uses the SAME unified prompts as RLM for fair comparison
# Only difference: context = top-k chunks (RAG) vs paragraph (RLM)
# ---------------------------------------------------------------------------

# Module-level hash verification (set on first init)
_PROMPT_HASH: Optional[str] = None


class RAGRunner:
    """
    Single-pass RAG runner for fair experiment comparison.
    
    Architecture:
    1. Retrieve top-k chunks using hybrid search
    2. Format chunks as context
    3. Single LLM call to extract obligations
    4. Return ActivityResponse + cost tracking
    """
    
    def __init__(
        self,
        model: str = MODEL,
        temperature: float = TEMPERATURE,
        retrieval_mode: str = "hybrid",
        top_k: int = 10,  # Number of chunks to retrieve
    ):
        global _PROMPT_HASH
        
        self.model = model
        self.temperature = temperature
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.client = OpenAI()
        self.retriever = Retriever(use_reranker=True)
        
        # Verify prompt hash on first init
        if _PROMPT_HASH is None:
            _PROMPT_HASH = verify_prompt_hash("RAG")
    
    def research(
        self,
        activity: str,
        states: list[str],
    ) -> tuple[ActivityResponse, QueryCost]:
        """
        Run single-pass RAG query.
        
        Args:
            activity: The regulatory activity (e.g., "incident reporting")
            states: List of state codes (usually just one)
        
        Returns:
            Tuple of (ActivityResponse, QueryCost)
        """
        cost = QueryCost()
        
        # Handle each state separately
        all_states = {}
        
        for state in states:
            state_result = self._query_state(activity, state, cost)
            if state_result:
                all_states[state] = state_result
        
        # Build ActivityResponse
        response = ActivityResponse(
            activity=activity,
            states=all_states,
        )
        
        return response, cost
    
    def _query_state(
        self,
        activity: str,
        state: str,
        cost: QueryCost,
    ) -> Optional[StateObligations]:
        """Query a single state and return StateObligations."""
        
        # Step 1: Retrieve relevant chunks
        query_text = f"{activity} requirements for {state}"
        
        # -----------------------------------------------------------------------
        # FAIRNESS FIX: Always include federal documents for state queries.
        # Federal rules (NERC CIP, DOE) apply to all utilities regardless of state.
        # This matches RLM behavior which always adds FED to target_states.
        # -----------------------------------------------------------------------
        if state == "FED":
            results = self.retriever.retrieve(
                query_text,
                states=["FED"],
                include_federal=True,
            )
        else:
            # Retrieve state-specific documents
            state_results = self.retriever.retrieve(
                query_text,
                states=[state],
                include_federal=False,
            )
            
            # Also retrieve federal documents (baseline applies to all states)
            fed_query = f"{activity} requirements federal"
            fed_results = self.retriever.retrieve(
                fed_query,
                states=["FED"],
                include_federal=True,
            )
            
            # Merge and dedupe (state results first, then federal)
            results = self._merge_results(state_results, fed_results)
        
        cost.add_retrieval()
        
        if not results:
            return StateObligations(
                obligations=[],
                confidence="low",
                not_found_explanation=f"No {state} documents found in corpus",
            )
        
        # Step 2: Format context (top-k chunks)
        context = self.retriever.format_context(results[:self.top_k])
        
        # ---------------------------------------------------------------------------
        # NO PRE-LLM GATING: Always call LLM when retrieval returns chunks.
        # The extraction prompt instructs the model to return empty obligations
        # if the context lacks explicit obligation language (Rule 2 in prompt).
        # ---------------------------------------------------------------------------
        
        # Step 3: Build prompt using SHARED templates
        user_prompt = EXTRACTION_USER_TEMPLATE.format(
            activity=activity,
            state=state,
            context=context,
        )
        
        # Step 4: Single LLM call
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=self.temperature,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            
            # Track cost
            cost.add_llm_call(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                call_type="extraction",
            )
            
            # Parse response - unified schema: {"obligations": [...]}
            raw_json = response.choices[0].message.content
            parsed = json.loads(raw_json)
            
            # Extract obligations from unified schema
            obligations = []
            for obl_data in parsed.get("obligations", []):
                # Build citations
                citations = []
                for cite_data in obl_data.get("citations", []):
                    citations.append(Citation(
                        doc_id=cite_data.get("doc_id", "unknown"),
                        page=cite_data.get("page"),
                        quote=cite_data.get("quote", ""),
                    ))
                
                obligations.append(Obligation(
                    obligation=obl_data.get("obligation"),
                    trigger=obl_data.get("trigger"),
                    deadline=obl_data.get("deadline"),
                    notify_who=obl_data.get("notify_who"),
                    required_content=obl_data.get("required_content"),
                    citations=citations,
                ))
            
            confidence = "high" if obligations else "low"
            return StateObligations(
                obligations=obligations,
                confidence=confidence,
                not_found_explanation=None if obligations else f"No {state} obligations found",
            )
                
        except Exception as e:
            return StateObligations(
                obligations=[],
                confidence="low",
                not_found_explanation=f"Error: {str(e)}",
            )

    def _merge_results(
        self,
        state_results: list[tuple],
        fed_results: list[tuple],
    ) -> list[tuple]:
        """
        Merge state and federal retrieval results, deduplicating by chunk_id.
        State results are prioritized (appear first).
        
        Args:
            state_results: List of (Chunk, score) tuples from state query
            fed_results: List of (Chunk, score) tuples from federal query
            
        Returns:
            Merged list with duplicates removed
        """
        seen_ids = set()
        merged = []
        
        # Add state results first (higher priority)
        for chunk, score in state_results:
            chunk_id = getattr(chunk, 'chunk_id', None) or id(chunk)
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                merged.append((chunk, score))
        
        # Add federal results (lower priority, dedupe)
        for chunk, score in fed_results:
            chunk_id = getattr(chunk, 'chunk_id', None) or id(chunk)
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                merged.append((chunk, score))
        
        return merged
