#!/usr/bin/env python3
"""
run_eval.py - Run evaluation harness against gold standard.

Usage:
    python -m eval.run_eval                    # Run all 20 queries
    python -m eval.run_eval --states NJ MD     # Run specific states
    python -m eval.run_eval --activities "incident reporting"  # Specific activity
    python -m eval.run_eval --dry-run          # Show queries without running
    python -m eval.run_eval --experiment rlm  # Run with specific experiment config

Output:
    - Console summary with all metrics
    - eval/results/eval_<timestamp>.json with detailed results

Experiment Configs:
    rag  - Single-pass RAG baseline (retrieve top-k, one LLM call)
    rlm  - RLM controller with iterative multi-doc search
"""

import sys
import os
import time
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env if present
try:
    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

from eval.evaluator import Evaluator, QueryResult
from eval.experiment_config import (
    EXPERIMENT_CONFIGS,
    QueryCost,
    MODEL as DEFAULT_MODEL,
    TEMPERATURE,
)
from eval.rag_runner import RAGRunner
from rlm.rlm_controller import RLMController
from rag.schemas import ActivityResponse


def parse_args():
    parser = argparse.ArgumentParser(description="Run RLM evaluation harness")
    parser.add_argument(
        "--states",
        nargs="+",
        default=None,
        help="States to evaluate (default: all)"
    )
    parser.add_argument(
        "--activities",
        nargs="+",
        default=None,
        help="Activities to evaluate (default: all)"
    )
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["hybrid", "bm25", "semantic"],
        help="Retrieval mode (default: hybrid)"
    )
    parser.add_argument(
        "--experiment",
        default=None,
        choices=list(EXPERIMENT_CONFIGS.keys()),
        help=f"Experiment config: {', '.join(EXPERIMENT_CONFIGS.keys())}"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show queries that would be run without executing"
    )
    parser.add_argument(
        "--gold",
        type=str,
        default=None,
        help="Path to gold standard JSON (default: eval/gold_standard.json)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval/results",
        help="Output directory for results (default: eval/results)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model (default: gpt-4o). Options: gpt-4o, gpt-4o-mini, etc."
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=None,
        help="Specific query IDs to run (default: all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed RLM extraction logs (default: condensed progress)"
    )
    return parser.parse_args()


def filter_queries(queries: list[dict], states: list[str] | None, activities: list[str] | None, query_ids: list[str] | None = None) -> list[dict]:
    """Filter queries by state, activity, and/or query ID."""
    filtered = queries
    
    # If specific query IDs are provided, use those directly
    if query_ids:
        filtered = [q for q in filtered if q["id"] in query_ids]
        return filtered
    
    if states:
        states_upper = [s.upper() for s in states]
        filtered = [q for q in filtered if q["state"].upper() in states_upper]
    
    if activities:
        activities_lower = [a.lower() for a in activities]
        filtered = [q for q in filtered if q["activity"].lower() in activities_lower]
    
    return filtered


def run_single_query_rlm(
    controller: RLMController,
    activity: str,
    state: str,
) -> tuple[ActivityResponse, QueryCost]:
    """
    Run a single query with RLM controller and return response with cost.
    
    Returns:
        Tuple of (ActivityResponse, QueryCost)
    """
    # Reset controller counters
    controller.iteration_count = 0
    controller.total_tool_calls = 0
    
    # Run research (returns tuple of response and cost)
    if state == "FED":
        response, cost = controller.research(activity, states=["FED"])
    else:
        response, cost = controller.research(activity, states=[state])
    
    return response, cost


def run_single_query_rag(
    runner: RAGRunner,
    activity: str,
    state: str,
) -> tuple[ActivityResponse, QueryCost]:
    """
    Run a single query with RAG runner and return response with cost.
    
    Returns:
        Tuple of (ActivityResponse, QueryCost)
    """
    if state == "FED":
        response, cost = runner.research(activity, states=["FED"])
    else:
        response, cost = runner.research(activity, states=[state])
    
    return response, cost

def main():
    args = parse_args()
    
    # Check API key
    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    
    # Model selection (can be overridden via --model flag)
    model = args.model or DEFAULT_MODEL
    
    # Load evaluator
    evaluator = Evaluator(gold_path=args.gold)
    queries = evaluator.get_gold_queries()
    
    # Filter queries
    queries = filter_queries(queries, args.states, args.activities, args.queries)
    
    if not queries:
        print("No queries match the filter criteria")
        sys.exit(1)
    
    print(f"Evaluation Configuration:")
    print(f"  Total queries: {len(queries)}")
    if args.queries:
        print(f"  Query IDs: {args.queries}")
    else:
        print(f"  States: {args.states or 'ALL'}")
        print(f"  Activities: {args.activities or 'ALL'}")
    print(f"  Model: {model}")
    print(f"  Temperature: {TEMPERATURE} (locked)")
    print("-" * 60)
    
    if args.dry_run:
        print("\n[DRY RUN] Queries that would be executed:")
        for i, q in enumerate(queries, 1):
            has_gold = "[+]" if q.get("must_catch") else "[o]"
            print(f"  {i:2}. {has_gold} {q['id']}: '{q['activity']}' in {q['state']}")
        print(f"\nTotal: {len(queries)} queries")
        print(f"  With gold obligation: {sum(1 for q in queries if q.get('must_catch'))}")
        print(f"  Without gold (expect null): {sum(1 for q in queries if not q.get('must_catch'))}")
        return
    
    # Determine experiment config
    if args.experiment:
        exp_config = EXPERIMENT_CONFIGS[args.experiment]
    else:
        # Default to rlm
        exp_config = EXPERIMENT_CONFIGS["rlm"]
    
    print(f"\n[Experiment: {exp_config.name}]")
    print(f"  Description: {exp_config.description}")
    print(f"  Use RAG: {exp_config.use_rag}")
    
    # Initialize runner based on experiment config
    controller = None
    rag_runner = None
    
    if exp_config.use_rag:
        print("\nInitializing RAG runner...")
        rag_runner = RAGRunner(
            model=model,
            temperature=TEMPERATURE,
            retrieval_mode=exp_config.retrieval_mode,
        )
    else:
        print("\nInitializing RLM controller...")
        controller = RLMController(
            model=model,
            max_iterations=12,
            retrieval_mode=exp_config.retrieval_mode,
            verbose=args.verbose,
        )
    
    # Run queries and track total cost
    results: list[QueryResult] = []
    total_cost = QueryCost()
    
    for i, query in enumerate(queries, 1):
        query_id = query["id"]
        activity = query["activity"]
        state = query["state"]
        has_gold = "[+]" if query.get("must_catch") else "[o]"
        
        print(f"\n[{i}/{len(queries)}] {has_gold} Running: {query_id}")
        print(f"  Activity: '{activity}' | State: {state}")
        
        start_time = time.time()
        
        try:
            # Dispatch to RAG or RLM based on config
            if exp_config.use_rag:
                response, cost = run_single_query_rag(rag_runner, activity, state)
            else:
                response, cost = run_single_query_rlm(controller, activity, state)
            
            elapsed = time.time() - start_time
            
            # Aggregate cost
            if cost:
                total_cost.llm_calls += cost.llm_calls
                total_cost.prompt_tokens += cost.prompt_tokens
                total_cost.completion_tokens += cost.completion_tokens
                total_cost.total_tokens += cost.total_tokens
                total_cost.tool_calls += cost.tool_calls
                # RLM verification metrics
                total_cost.rlm_obligations_verified += cost.rlm_obligations_verified
                total_cost.rlm_obligations_dropped_by_verifier += cost.rlm_obligations_dropped_by_verifier
                total_cost.rlm_repair_attempts += cost.rlm_repair_attempts
                total_cost.rlm_fields_repaired_deadline += cost.rlm_fields_repaired_deadline
                total_cost.rlm_fields_repaired_notify += cost.rlm_fields_repaired_notify
                total_cost.rlm_repair_rejected_deadline += cost.rlm_repair_rejected_deadline
                total_cost.rlm_repair_rejected_notify += cost.rlm_repair_rejected_notify
            
            # Evaluate
            result = evaluator.evaluate_single(
                query=query,
                response=response,
                tool_calls=cost.tool_calls if cost else 0,
                elapsed=elapsed,
            )
            # Add cost details to result
            if cost:
                result.rlm_obligations_verified = cost.rlm_obligations_verified
                result.rlm_obligations_dropped_by_verifier = cost.rlm_obligations_dropped_by_verifier
                result.rlm_repair_attempts = cost.rlm_repair_attempts
                result.rlm_fields_repaired_deadline = cost.rlm_fields_repaired_deadline
                result.rlm_fields_repaired_notify = cost.rlm_fields_repaired_notify
                result.rlm_repair_rejected_deadline = cost.rlm_repair_rejected_deadline
                result.rlm_repair_rejected_notify = cost.rlm_repair_rejected_notify
                result.llm_tokens = cost.total_tokens
            
            results.append(result)
            
            # Print immediate feedback
            status = "MISS" if result.critical_miss else ("FOUND" if result.has_gold_obligation else "OK")
            extras = ""
            if cost:
                # Show verification stats for RLM
                if cost.rlm_obligations_verified > 0:
                    extras += f" V:{cost.rlm_obligations_verified}"
                if cost.rlm_obligations_dropped_by_verifier > 0:
                    extras += f" D:{cost.rlm_obligations_dropped_by_verifier}"
                if cost.rlm_repair_attempts > 0:
                    extras += f" R:{cost.rlm_repair_attempts}"
                extras += f" [{cost.llm_calls}LLM/{cost.total_tokens}tok]"
            print(f"  -> {status} | {len(result.found_obligations)} obligations | "
                  f"{result.citation_valid_count}/{result.citation_total_count} valid cites | "
                  f"{elapsed:.1f}s{extras}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            # Create a failed result
            result = QueryResult(
                query_id=query_id,
                activity=activity,
                state=state,
                has_gold_obligation=query.get("must_catch") is not None,
                gold_obligation=query.get("must_catch"),
                critical_miss=True if query.get("must_catch") else False,
                elapsed_seconds=time.time() - start_time,
            )
            results.append(result)
    
    # Compute summary
    summary = evaluator.summarize(results)
    evaluator.print_summary(summary)
    
    # Print cost summary (Regime A)
    print("\n" + "=" * 60)
    print("[Regime A: Cost Comparison]")
    print(f"  Total LLM calls: {total_cost.llm_calls}")
    print(f"  Total tokens: {total_cost.total_tokens:,}")
    print(f"    Prompt: {total_cost.prompt_tokens:,}")
    print(f"    Completion: {total_cost.completion_tokens:,}")
    print(f"  Estimated cost: ${total_cost.to_dict()['estimated_cost_usd']:.4f}")
    print(f"  RLM verified: {total_cost.rlm_obligations_verified}")
    print(f"  RLM dropped by verifier: {total_cost.rlm_obligations_dropped_by_verifier}")
    print(f"  RLM repairs attempted: {total_cost.rlm_repair_attempts}")
    print("=" * 60)
    
    # Print error analysis
    evaluator.print_error_analysis(results)
    
    # Save results with experiment name in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = exp_config.name
    output_path = Path(args.output_dir) / f"eval_{exp_name}_{timestamp}.json"
    
    # Add cost data to saved results
    evaluator.save_results(
        results, 
        summary, 
        output_path,
        extra_data={
            "experiment": exp_config.name,
            "use_rag": exp_config.use_rag,
            "model": model,
            "temperature": TEMPERATURE,
            "cost": total_cost.to_dict(),
        }
    )
    
    # Print experiment config in results
    print(f"\n[Experiment: {exp_config.name}]")
    print(f"  Config: use_rag={exp_config.use_rag}")
    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
