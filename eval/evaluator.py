"""
evaluator.py
------------
Evaluation harness for RLM system.

Metrics:
1. Critical Miss Rate (headline) - Did we fail to find must-catch obligations?
2. Citation Validity - Do citations exist in corpus?
3. Field Accuracy - Do deadline/notify_who match gold standard?
4. False Obligation Rate - How many obligations are hallucinated?
5. Cost/Time - Tool calls + LLM tokens
6. Error Analysis - Categorized failure modes
"""

import json
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.schemas import ActivityResponse, Obligation


# ---------------------------------------------------------------------------
# Error Analysis Categories
# ---------------------------------------------------------------------------
class ErrorCategory(str, Enum):
    """Categories of errors for structured analysis."""
    DOC_NOT_DISCOVERED = "doc_not_discovered"  # Correct doc not retrieved
    WRONG_SPAN = "wrong_span"  # Correct doc but wrong section/span
    EXTRACTOR_MISSED_FIELD = "extractor_missed_field"  # Field exists but not extracted
    FIELD_IN_OTHER_PARAGRAPH = "field_in_other_paragraph"  # Field exists in nearby paragraph
    CITE_DROPPED = "cite_dropped"  # Citation not included in output
    ROUTING_VIOLATION = "routing_violation"  # State query cites federal doc
    FALSE_POSITIVE = "false_positive"  # Obligation not in gold standard
    SUCCESS = "success"  # Correctly found obligation


@dataclass
class ErrorRecord:
    """Single error record for analysis."""
    query_id: str
    state: str
    activity: str
    category: str
    gold_doc_id: Optional[str] = None
    found_doc_ids: list[str] = field(default_factory=list)
    gold_deadline: Optional[str] = None
    found_deadline: Optional[str] = None
    gold_notify_who: Optional[str] = None
    found_notify_who: Optional[str] = None
    details: Optional[str] = None


@dataclass
class ErrorAnalysis:
    """Aggregated error analysis across all queries."""
    total_queries: int = 0
    successes: int = 0
    errors_by_category: dict = field(default_factory=dict)
    error_records: list[ErrorRecord] = field(default_factory=list)
    
    def add_error(self, record: ErrorRecord) -> None:
        """Add an error record and update category counts."""
        self.error_records.append(record)
        cat = record.category
        self.errors_by_category[cat] = self.errors_by_category.get(cat, 0) + 1
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_queries": self.total_queries,
            "successes": self.successes,
            "success_rate": self.successes / self.total_queries if self.total_queries > 0 else 0,
            "errors_by_category": self.errors_by_category,
            "error_records": [asdict(r) for r in self.error_records],
        }


@dataclass
class QueryResult:
    """Result from running a single query."""
    query_id: str
    activity: str
    state: str
    # Ground truth
    has_gold_obligation: bool
    gold_obligation: Optional[dict] = None
    expected_no_obligation: bool = False  # True for negative test cases
    # System output
    found_obligations: list[Obligation] = field(default_factory=list)
    # Metrics
    critical_miss: bool = False  # True if gold exists but not found
    gold_match_position: Optional[int] = None  # Index of matched obligation (0 = top-1)
    false_positive_count: int = 0
    negative_test_passed: Optional[bool] = None  # True if correctly returned no obligation
    citation_valid_count: int = 0
    citation_total_count: int = 0
    deadline_match: Optional[bool] = None  # None if no gold deadline
    notify_who_match: Optional[bool] = None  # None if no gold notify_who
    routing_violation: bool = False  # True if state query cites FED doc
    routing_violation_docs: list[str] = field(default_factory=list)  # Which FED docs were cited
    # Error analysis
    error_category: Optional[str] = None  # Categorized error type
    error_details: Optional[str] = None  # Additional error context
    # Cost
    tool_calls: int = 0
    llm_tokens: int = 0
    elapsed_seconds: float = 0.0
    
    # RLM Verification & Repair metrics
    rlm_obligations_verified: int = 0  # Obligations verified
    rlm_obligations_dropped_by_verifier: int = 0  # Dropped due to activity mismatch
    rlm_repair_attempts: int = 0  # Repair attempts made
    rlm_fields_repaired_deadline: int = 0  # Deadlines successfully repaired
    rlm_fields_repaired_notify: int = 0  # Notify_who successfully repaired
    rlm_repair_rejected_deadline: int = 0  # Deadline repairs rejected
    rlm_repair_rejected_notify: int = 0  # Notify repairs rejected
    
    @property
    def found_match(self) -> bool:
        """Did we find any obligation that matched the gold obligation?"""
        return self.has_gold_obligation and not self.critical_miss
    
    @property
    def citation_validity_rate(self) -> float:
        if self.citation_total_count == 0:
            return 1.0
        return self.citation_valid_count / self.citation_total_count


@dataclass
class EvalSummary:
    """Summary statistics across all queries."""
    total_queries: int = 0
    queries_with_gold: int = 0
    queries_without_gold: int = 0
    
    # Negative test cases
    negative_tests: int = 0
    negative_tests_passed: int = 0
    negative_test_accuracy: float = 0.0
    
    # Critical miss
    critical_misses: int = 0
    critical_miss_rate: float = 0.0
    
    # Top-1 hit rate (gold found at position 0)
    top1_hits: int = 0
    top1_hit_rate: float = 0.0
    
    # Citation validity
    valid_citations: int = 0
    total_citations: int = 0
    citation_validity_rate: float = 0.0
    
    # Field accuracy
    deadline_matches: int = 0
    deadline_total: int = 0
    deadline_accuracy: float = 0.0
    notify_who_matches: int = 0
    notify_who_total: int = 0
    notify_who_accuracy: float = 0.0
    
    # False obligations
    false_obligations: int = 0
    total_obligations_found: int = 0
    false_obligation_rate: float = 0.0
    
    # Routing violations (state queries citing FED docs)
    routing_violations: int = 0
    routing_violation_rate: float = 0.0
    
    # Cost
    total_tool_calls: int = 0
    total_llm_tokens: int = 0
    total_elapsed_seconds: float = 0.0
    avg_tool_calls_per_query: float = 0.0
    avg_elapsed_per_query: float = 0.0
    
    # RLM Verification & Repair summary
    total_rlm_verified: int = 0  # Obligations verified
    total_rlm_dropped: int = 0  # Dropped due to activity mismatch
    total_repair_attempts: int = 0  # Repair attempts made
    total_repaired_deadline: int = 0  # Deadlines successfully repaired
    total_repaired_notify: int = 0  # Notify_who successfully repaired
    total_rejected_deadline: int = 0  # Deadline repairs rejected
    total_rejected_notify: int = 0  # Notify repairs rejected
    
    # Breakdown by state
    results_by_state: dict = field(default_factory=dict)
    
    def compute(self, results: list[QueryResult]) -> None:
        """Compute summary statistics from individual results."""
        self.total_queries = len(results)
        self.queries_with_gold = sum(1 for r in results if r.has_gold_obligation)
        self.queries_without_gold = self.total_queries - self.queries_with_gold
        
        # Negative tests (expected_no_obligation)
        negative_results = [r for r in results if r.expected_no_obligation]
        self.negative_tests = len(negative_results)
        self.negative_tests_passed = sum(1 for r in negative_results if r.negative_test_passed)
        if self.negative_tests > 0:
            self.negative_test_accuracy = self.negative_tests_passed / self.negative_tests
        
        # Critical miss (only count queries that have gold, excluding negative tests)
        positive_results = [r for r in results if r.has_gold_obligation]
        self.critical_misses = sum(1 for r in positive_results if r.critical_miss)
        if len(positive_results) > 0:
            self.critical_miss_rate = self.critical_misses / len(positive_results)
        
        # Top-1 hit rate (gold found at position 0)
        self.top1_hits = sum(1 for r in positive_results if r.gold_match_position == 0)
        if len(positive_results) > 0:
            self.top1_hit_rate = self.top1_hits / len(positive_results)
        
        # Citation validity
        self.valid_citations = sum(r.citation_valid_count for r in results)
        self.total_citations = sum(r.citation_total_count for r in results)
        if self.total_citations > 0:
            self.citation_validity_rate = self.valid_citations / self.total_citations
        
        # Field accuracy
        for r in results:
            if r.deadline_match is not None:
                self.deadline_total += 1
                if r.deadline_match:
                    self.deadline_matches += 1
            if r.notify_who_match is not None:
                self.notify_who_total += 1
                if r.notify_who_match:
                    self.notify_who_matches += 1
        
        if self.deadline_total > 0:
            self.deadline_accuracy = self.deadline_matches / self.deadline_total
        if self.notify_who_total > 0:
            self.notify_who_accuracy = self.notify_who_matches / self.notify_who_total
        
        # False obligations
        self.false_obligations = sum(r.false_positive_count for r in results)
        self.total_obligations_found = sum(len(r.found_obligations) for r in results)
        if self.total_obligations_found > 0:
            self.false_obligation_rate = self.false_obligations / self.total_obligations_found
        
        # Routing violations (state queries citing FED docs)
        state_queries = [r for r in results if r.state != "FED"]
        self.routing_violations = sum(1 for r in state_queries if r.routing_violation)
        if state_queries:
            self.routing_violation_rate = self.routing_violations / len(state_queries)
        
        # Cost
        self.total_tool_calls = sum(r.tool_calls for r in results)
        self.total_llm_tokens = sum(r.llm_tokens for r in results)
        self.total_elapsed_seconds = sum(r.elapsed_seconds for r in results)
        if self.total_queries > 0:
            self.avg_tool_calls_per_query = self.total_tool_calls / self.total_queries
            self.avg_elapsed_per_query = self.total_elapsed_seconds / self.total_queries
        
        # RLM Verification & Repair
        self.total_rlm_verified = sum(r.rlm_obligations_verified for r in results)
        self.total_rlm_dropped = sum(r.rlm_obligations_dropped_by_verifier for r in results)
        self.total_repair_attempts = sum(r.rlm_repair_attempts for r in results)
        self.total_repaired_deadline = sum(r.rlm_fields_repaired_deadline for r in results)
        self.total_repaired_notify = sum(r.rlm_fields_repaired_notify for r in results)
        self.total_rejected_deadline = sum(r.rlm_repair_rejected_deadline for r in results)
        self.total_rejected_notify = sum(r.rlm_repair_rejected_notify for r in results)
        
        # By state breakdown
        states = set(r.state for r in results)
        for state in states:
            state_results = [r for r in results if r.state == state]
            self.results_by_state[state] = {
                "queries": len(state_results),
                "with_gold": sum(1 for r in state_results if r.has_gold_obligation),
                "critical_misses": sum(1 for r in state_results if r.critical_miss),
                "obligations_found": sum(len(r.found_obligations) for r in state_results),
            }


class Evaluator:
    """
    Run evaluation against gold standard.
    """
    
    def __init__(
        self,
        gold_path: str | Path = None,
        corpus_path: str | Path = None,
    ):
        self.gold_path = Path(gold_path) if gold_path else Path(__file__).parent / "gold_standard.json"
        self.corpus_path = Path(corpus_path) if corpus_path else Path(__file__).parent.parent / "corpus" / "normalized"
        
        # Load gold standard
        with open(self.gold_path) as f:
            self.gold_data = json.load(f)
        
        # Load valid doc_ids from corpus
        self.valid_doc_ids = self._load_valid_doc_ids()
    
    def _load_valid_doc_ids(self) -> set[str]:
        """Load all valid document IDs from corpus."""
        doc_ids = set()
        meta_dir = self.corpus_path / "doc_meta"
        if meta_dir.exists():
            for f in meta_dir.glob("*.json"):
                with open(f) as fp:
                    meta = json.load(fp)
                doc_ids.add(meta.get("doc_id", ""))
        return doc_ids
    
    def get_gold_queries(self) -> list[dict]:
        """Return all gold standard queries."""
        return self.gold_data.get("queries", [])
    
    def evaluate_single(
        self,
        query: dict,
        response: ActivityResponse,
        tool_calls: int = 0,
        llm_tokens: int = 0,
        elapsed: float = 0.0,
    ) -> QueryResult:
        """
        Evaluate a single query against gold standard.
        
        Args:
            query: Gold standard query dict
            response: ActivityResponse from RLM
            tool_calls: Number of tool calls made
            llm_tokens: LLM tokens used
            elapsed: Wall-clock time in seconds
        """
        # Check if this is a negative test case
        expected_no_obligation = query.get("expected_result") == "no_obligation"
        
        result = QueryResult(
            query_id=query["id"],
            activity=query["activity"],
            state=query["state"],
            has_gold_obligation=query.get("must_catch") is not None,
            gold_obligation=query.get("must_catch"),
            expected_no_obligation=expected_no_obligation,
            tool_calls=tool_calls,
            llm_tokens=llm_tokens,
            elapsed_seconds=elapsed,
        )
        
        # Get state obligations from response
        state = query["state"]
        if state == "FED":
            state_obs = response.federal_baseline
        else:
            state_obs = response.states.get(state)
        
        if state_obs and state_obs.obligations:
            result.found_obligations = list(state_obs.obligations)
        
        # For state queries, also include federal baseline obligations
        # (federal requirements apply to all regulated utilities)
        if state != "FED" and response.federal_baseline and response.federal_baseline.obligations:
            result.found_obligations = result.found_obligations + list(response.federal_baseline.obligations)
        
        # Check citation validity
        for ob in result.found_obligations:
            for cit in ob.citations:
                result.citation_total_count += 1
                if cit.doc_id in self.valid_doc_ids:
                    result.citation_valid_count += 1
        
        # Check for routing violations (state queries citing FED docs)
        # NOTE: FED docs are now VALID for state queries (federal baseline applies to all states)
        # We still track which FED docs were cited for transparency, but it's not a violation
        if state != "FED":
            for ob in result.found_obligations:
                for cit in ob.citations:
                    if cit.doc_id.startswith("FED_"):
                        # Track FED docs cited (informational, not a violation)
                        if cit.doc_id not in result.routing_violation_docs:
                            result.routing_violation_docs.append(cit.doc_id)
        # routing_violation stays False - FED docs are valid for state queries
        
        # Handle negative test cases (expected to find no obligation)
        if result.expected_no_obligation:
            if len(result.found_obligations) == 0:
                result.negative_test_passed = True
                result.error_category = ErrorCategory.SUCCESS.value
            else:
                result.negative_test_passed = False
                result.false_positive_count = len(result.found_obligations)
                result.error_category = ErrorCategory.FALSE_POSITIVE.value
                result.error_details = f"Expected no obligation but found {len(result.found_obligations)}"
            return result
        
        # If no gold obligation, any obligation found is potentially false
        if not result.has_gold_obligation:
            result.false_positive_count = len(result.found_obligations)
            return result
        
        # Check if we found the must-catch obligation
        gold = result.gold_obligation
        # Support multiple valid doc_ids, fallback to single cite.doc_id
        valid_doc_ids = set(gold.get("valid_doc_ids", [gold["cite"]["doc_id"]]))
        
        matched = False
        matched_idx = None
        for idx, ob in enumerate(result.found_obligations):
            # Check if any citation points to a valid doc
            for cit in ob.citations:
                if cit.doc_id in valid_doc_ids:
                    matched = True
                    matched_idx = idx
                    
                    # Check field accuracy
                    if gold.get("deadline"):
                        result.deadline_match = self._fuzzy_match(
                            gold["deadline"], 
                            ob.deadline or ""
                        )
                    
                    if gold.get("notify_who"):
                        result.notify_who_match = self._fuzzy_match(
                            gold["notify_who"],
                            ob.notify_who or ""
                        )
                    break
            if matched:
                break
        
        result.critical_miss = not matched
        result.gold_match_position = matched_idx
        
        # False positives: obligations that don't cite any valid doc
        # For state queries, FED docs are also valid (federal baseline applies)
        # So we only count as false positive if doc is neither in valid_doc_ids nor a FED doc
        def is_valid_citation(cit, state):
            if cit.doc_id in valid_doc_ids:
                return True
            # For state queries, FED docs are also valid
            if state != "FED" and cit.doc_id.startswith("FED_") and cit.doc_id in self.valid_doc_ids:
                return True
            return False
        
        result.false_positive_count = sum(
            1 for ob in result.found_obligations
            if not any(is_valid_citation(cit, state) for cit in ob.citations)
        )
        
        # Note: RLM verification fields are set by run_eval.py from QueryCost
        
        # Error analysis: categorize the error
        result = self._categorize_error(result, gold, valid_doc_ids)
        
        return result
    
    def _categorize_error(
        self,
        result: QueryResult,
        gold: dict,
        valid_doc_ids: set[str],
    ) -> QueryResult:
        """Categorize the error type for structured analysis."""
        
        # Check for routing violation first
        if result.routing_violation:
            result.error_category = ErrorCategory.ROUTING_VIOLATION.value
            result.error_details = f"State query cited FED docs: {result.routing_violation_docs}"
            return result
        
        # Success case
        if not result.critical_miss:
            if result.deadline_match is False or result.notify_who_match is False:
                result.error_category = ErrorCategory.EXTRACTOR_MISSED_FIELD.value
                missed = []
                if result.deadline_match is False:
                    missed.append(f"deadline (gold: {gold.get('deadline')})")
                if result.notify_who_match is False:
                    missed.append(f"notify_who (gold: {gold.get('notify_who')})")
                result.error_details = f"Matched doc but missed fields: {', '.join(missed)}"
            else:
                result.error_category = ErrorCategory.SUCCESS.value
            return result
        
        # Critical miss - need to determine why
        found_doc_ids = set()
        for ob in result.found_obligations:
            for cit in ob.citations:
                found_doc_ids.add(cit.doc_id)
        
        # Check if we found any docs at all
        if not found_doc_ids:
            result.error_category = ErrorCategory.DOC_NOT_DISCOVERED.value
            result.error_details = f"No obligations found. Expected docs: {valid_doc_ids}"
            return result
        
        # Check if we found any valid docs but wrong section
        if found_doc_ids & valid_doc_ids:
            result.error_category = ErrorCategory.WRONG_SPAN.value
            result.error_details = f"Found correct doc(s) {found_doc_ids & valid_doc_ids} but wrong section/obligation"
            return result
        
        # Found wrong docs entirely
        result.error_category = ErrorCategory.DOC_NOT_DISCOVERED.value
        result.error_details = f"Found docs {found_doc_ids} but needed {valid_doc_ids}"
        return result
    
    def _fuzzy_match(self, gold: str, found: str) -> bool:
        """Fuzzy match for deadline/notify_who fields."""
        if not gold or not found:
            return False
        
        # Normalize
        gold_norm = re.sub(r'\s+', ' ', gold.lower().strip())
        found_norm = re.sub(r'\s+', ' ', found.lower().strip())
        
        # Exact match
        if gold_norm == found_norm:
            return True
        
        # Substring match (gold contained in found or vice versa)
        if gold_norm in found_norm or found_norm in gold_norm:
            return True
        
        # Extract numbers and check
        gold_nums = set(re.findall(r'\d+', gold))
        found_nums = set(re.findall(r'\d+', found))
        if gold_nums and gold_nums == found_nums:
            # Same numbers, likely same deadline
            return True
        
        # Key terms match
        gold_terms = set(gold_norm.split())
        found_terms = set(found_norm.split())
        overlap = gold_terms & found_terms
        if len(overlap) >= min(2, len(gold_terms)):
            return True
        
        return False
    
    def summarize(self, results: list[QueryResult]) -> EvalSummary:
        """Compute summary statistics from results."""
        summary = EvalSummary()
        summary.compute(results)
        return summary
    
    def print_summary(self, summary: EvalSummary) -> None:
        """Print formatted summary."""
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        
        print(f"\nTotal Queries: {summary.total_queries}")
        print(f"  Positive tests (obligation expected): {summary.queries_with_gold}")
        print(f"  Negative tests (no obligation expected): {summary.negative_tests}")
        
        print(f"\n{'=' * 40}")
        print("CRITICAL MISS RATE (headline)")
        print(f"{'=' * 40}")
        print(f"  Misses: {summary.critical_misses} / {summary.queries_with_gold}")
        print(f"  Rate: {summary.critical_miss_rate:.1%}")
        
        print(f"\n{'=' * 40}")
        print("TOP-1 HIT RATE (ranking quality)")
        print(f"{'=' * 40}")
        print(f"  Top-1 Hits: {summary.top1_hits} / {summary.queries_with_gold}")
        print(f"  Rate: {summary.top1_hit_rate:.1%}")
        
        if summary.negative_tests > 0:
            print(f"\n{'=' * 40}")
            print("NEGATIVE TEST ACCURACY")
            print(f"{'=' * 40}")
            print(f"  Passed: {summary.negative_tests_passed} / {summary.negative_tests}")
            print(f"  Rate: {summary.negative_test_accuracy:.1%}")
        
        print(f"\n{'=' * 40}")
        print("CITATION VALIDITY")
        print(f"{'=' * 40}")
        print(f"  Valid: {summary.valid_citations} / {summary.total_citations}")
        print(f"  Rate: {summary.citation_validity_rate:.1%}")
        
        print(f"\n{'=' * 40}")
        print("FIELD ACCURACY")
        print(f"{'=' * 40}")
        print(f"  Deadline: {summary.deadline_matches} / {summary.deadline_total} "
              f"({summary.deadline_accuracy:.1%})")
        print(f"  Notify Who: {summary.notify_who_matches} / {summary.notify_who_total} "
              f"({summary.notify_who_accuracy:.1%})")
        
        print(f"\n{'=' * 40}")
        print("FALSE OBLIGATION RATE")
        print(f"{'=' * 40}")
        print(f"  False: {summary.false_obligations} / {summary.total_obligations_found}")
        print(f"  Rate: {summary.false_obligation_rate:.1%}")
        
        print(f"\n{'=' * 40}")
        print("ROUTING VERIFICATION")
        print(f"{'=' * 40}")
        if summary.routing_violations == 0:
            print(f"  [OK] No routing violations (state queries correctly separated from FED)")
        else:
            print(f"  [X] Violations: {summary.routing_violations} state queries cite FED docs")
            print(f"  Rate: {summary.routing_violation_rate:.1%}")
        
        print(f"\n{'=' * 40}")
        print("COST / TIME")
        print(f"{'=' * 40}")
        print(f"  Total tool calls: {summary.total_tool_calls}")
        print(f"  Total LLM tokens: {summary.total_llm_tokens:,}")
        if summary.total_rlm_verified > 0 or summary.total_rlm_dropped > 0:
            print(f"  RLM verified: {summary.total_rlm_verified}")
            print(f"  RLM dropped: {summary.total_rlm_dropped}")
            print(f"  Repair attempts: {summary.total_repair_attempts}")
            print(f"    Repaired deadline: {summary.total_repaired_deadline}")
            print(f"    Repaired notify: {summary.total_repaired_notify}")
            print(f"    Rejected deadline: {summary.total_rejected_deadline}")
            print(f"    Rejected notify: {summary.total_rejected_notify}")
        print(f"  Avg tool calls per query: {summary.avg_tool_calls_per_query:.1f}")
        print(f"  Total time: {summary.total_elapsed_seconds:.1f}s")
        print(f"  Avg per query: {summary.avg_elapsed_per_query:.1f}s")
        
        print(f"\n{'=' * 40}")
        print("BY STATE")
        print(f"{'=' * 40}")
        for state, stats in sorted(summary.results_by_state.items()):
            miss_rate = ""
            if stats["with_gold"] > 0:
                miss_rate = f" (miss rate: {stats['critical_misses']/stats['with_gold']:.0%})"
            print(f"  {state}: {stats['queries']} queries, "
                  f"{stats['obligations_found']} obligations found{miss_rate}")
    
    def print_error_analysis(self, results: list[QueryResult]) -> None:
        """Print error analysis summary."""
        analysis = self.generate_error_analysis(results)
        
        print(f"\n{'=' * 40}")
        print("ERROR ANALYSIS")
        print(f"{'=' * 40}")
        
        if analysis.total_queries == 0:
            print("  No queries with gold obligations")
            return
        
        # Print error categories
        print(f"  Success rate: {analysis.successes}/{analysis.total_queries} "
              f"({100*analysis.successes/analysis.total_queries:.1f}%)")
        print(f"\n  Errors by category:")
        for cat, count in sorted(analysis.errors_by_category.items()):
            if cat != ErrorCategory.SUCCESS.value:
                print(f"    {cat}: {count}")
        
        # Show top errors
        errors = [r for r in analysis.error_records if r.category != ErrorCategory.SUCCESS.value]
        if errors:
            print(f"\n  Top errors:")
            for e in errors[:5]:
                print(f"    - [{e.state}] {e.query_id}: {e.category}")
                if e.details:
                    print(f"      {e.details[:80]}...")
    
    def generate_error_analysis(self, results: list[QueryResult]) -> ErrorAnalysis:
        """Generate structured error analysis from results."""
        analysis = ErrorAnalysis(
            total_queries=len(results),
        )
        
        for r in results:
            if not r.has_gold_obligation:
                # Skip queries without gold - no error analysis needed
                continue
            
            # Create error record
            gold = r.gold_obligation
            valid_doc_ids = gold.get("valid_doc_ids", [gold["cite"]["doc_id"]]) if gold else []
            
            found_doc_ids = []
            found_deadline = None
            found_notify_who = None
            
            for ob in r.found_obligations:
                for cit in ob.citations:
                    found_doc_ids.append(cit.doc_id)
                if ob.deadline:
                    found_deadline = ob.deadline
                if ob.notify_who:
                    found_notify_who = ob.notify_who
            
            record = ErrorRecord(
                query_id=r.query_id,
                state=r.state,
                activity=r.activity,
                category=r.error_category or ErrorCategory.DOC_NOT_DISCOVERED.value,
                gold_doc_id=valid_doc_ids[0] if valid_doc_ids else None,
                found_doc_ids=list(set(found_doc_ids)),
                gold_deadline=gold.get("deadline") if gold else None,
                found_deadline=found_deadline,
                gold_notify_who=gold.get("notify_who") if gold else None,
                found_notify_who=found_notify_who,
                details=r.error_details,
            )
            
            if r.error_category == ErrorCategory.SUCCESS.value:
                analysis.successes += 1
            
            analysis.add_error(record)
        
        return analysis
    
    def save_results(
        self,
        results: list[QueryResult],
        summary: EvalSummary,
        output_path: str | Path,
        extra_data: dict | None = None,
    ) -> None:
        """Save detailed results to JSON."""
        # Generate error analysis
        error_analysis = self.generate_error_analysis(results)
        
        output = {
            "summary": {
                "total_queries": summary.total_queries,
                "queries_with_gold": summary.queries_with_gold,
                "negative_tests": summary.negative_tests,
                "negative_tests_passed": summary.negative_tests_passed,
                "negative_test_accuracy": summary.negative_test_accuracy,
                "critical_miss_rate": summary.critical_miss_rate,
                "citation_validity_rate": summary.citation_validity_rate,
                "deadline_accuracy": summary.deadline_accuracy,
                "notify_who_accuracy": summary.notify_who_accuracy,
                "false_obligation_rate": summary.false_obligation_rate,
                "routing_violations": summary.routing_violations,
                "routing_violation_rate": summary.routing_violation_rate,
                "total_tool_calls": summary.total_tool_calls,
                "total_llm_tokens": summary.total_llm_tokens,
                "total_rlm_verified": summary.total_rlm_verified,
                "total_rlm_dropped": summary.total_rlm_dropped,
                "total_repair_attempts": summary.total_repair_attempts,
                "total_repaired_deadline": summary.total_repaired_deadline,
                "total_repaired_notify": summary.total_repaired_notify,
                "total_rejected_deadline": summary.total_rejected_deadline,
                "total_rejected_notify": summary.total_rejected_notify,
                "total_elapsed_seconds": summary.total_elapsed_seconds,
                "avg_tool_calls_per_query": summary.avg_tool_calls_per_query,
                "results_by_state": summary.results_by_state,
            },
            "error_analysis": error_analysis.to_dict(),
            "by_state": summary.results_by_state,
            "results": [
                {
                    "query_id": r.query_id,
                    "activity": r.activity,
                    "state": r.state,
                    "has_gold": r.has_gold_obligation,
                    "expected_no_obligation": r.expected_no_obligation,
                    "negative_test_passed": r.negative_test_passed,
                    "critical_miss": r.critical_miss,
                    "error_category": r.error_category,
                    "error_details": r.error_details,
                    "obligations_found": len(r.found_obligations),
                    "citation_valid_rate": r.citation_validity_rate,
                    "deadline_match": r.deadline_match,
                    "notify_who_match": r.notify_who_match,
                    "false_positives": r.false_positive_count,
                    "routing_violation": r.routing_violation,
                    "tool_calls": r.tool_calls,
                    "rlm_obligations_verified": r.rlm_obligations_verified,
                    "rlm_obligations_dropped_by_verifier": r.rlm_obligations_dropped_by_verifier,
                    "rlm_repair_attempts": r.rlm_repair_attempts,
                    "rlm_fields_repaired_deadline": r.rlm_fields_repaired_deadline,
                    "rlm_fields_repaired_notify": r.rlm_fields_repaired_notify,
                    "rlm_repair_rejected_deadline": r.rlm_repair_rejected_deadline,
                    "rlm_repair_rejected_notify": r.rlm_repair_rejected_notify,
                    "llm_tokens": r.llm_tokens,
                    "elapsed_seconds": r.elapsed_seconds,
                }
                for r in results
            ],
        }
        
        # Add extra data (experiment config, cost) if provided
        if extra_data:
            output["experiment"] = extra_data
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n[Saved detailed results to {output_path}]")
