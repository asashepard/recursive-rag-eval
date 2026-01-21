#!/usr/bin/env python3
"""
run_rlm.py - Run the RLM controller for regulatory research.

Usage:
    python run_rlm.py "cybersecurity incident reporting" TX MD GA
    python run_rlm.py "breach notification timeline" ALL
    python run_rlm.py "incident reporting" TX --mode=bm25
    python run_rlm.py "incident reporting" TX --mode=semantic
    python run_rlm.py "incident reporting" TX --model gpt-4o-mini
    
Retrieval modes:
    --mode=hybrid   (default) Semantic + BM25 combined
    --mode=bm25     BM25 only (faster, keyword-focused)
    --mode=semantic Semantic only (embedding-based)
    
Environment:
    OPENAI_API_KEY must be set (or in .env file)
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env if present
try:
    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment from {env_path}")
except ImportError:
    pass  # dotenv not installed, rely on environment

# Verify API key
if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not set")
    print("Set it via environment variable or create a .env file")
    sys.exit(1)

from rlm.rlm_controller import RLMController


def parse_args():
    parser = argparse.ArgumentParser(description="Run the RLM controller for ad-hoc research")
    parser.add_argument("activity", type=str, help="Research activity, e.g. 'incident reporting'")
    parser.add_argument(
        "states",
        nargs="*",
        default=None,
        help="States to query (e.g. TX MD GA). Use ALL or omit for all states.",
    )
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["hybrid", "bm25", "semantic"],
        help="Retrieval mode (default: hybrid)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model name (default: gpt-4o)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=12,
        help="Global budget cap for controller iterations (default: 12)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write full JSON output",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    activity = args.activity
    retrieval_mode = args.mode
    states = args.states

    if not states or (len(states) == 1 and states[0].upper() == "ALL"):
        states = None
    
    print(f"Activity: {activity}")
    print(f"States: {states if states else 'ALL'}")
    print(f"Retrieval mode: {retrieval_mode}")
    print(f"Model: {args.model}")
    print(f"Max iterations: {args.max_iterations}")
    print("-" * 60)
    
    # RLM v2: use max_iterations with retrieval_mode
    controller = RLMController(
        model=args.model,
        max_iterations=args.max_iterations,
        retrieval_mode=retrieval_mode,
    )
    result, cost = controller.research(activity, states)
    
    print("\n" + "=" * 60)
    print("RESEARCH RESULTS")
    print("=" * 60)
    
    # Federal baseline
    if result.federal_baseline and result.federal_baseline.obligations:
        print(f"\n{'=' * 40}")
        print("FEDERAL BASELINE (NERC/DOE)")
        print(f"{'=' * 40}")
        for i, ob in enumerate(result.federal_baseline.obligations, 1):
            print(f"\n  [{i}] {ob.obligation}")
            if ob.deadline:
                print(f"      Deadline: {ob.deadline}")
            if ob.citations:
                for cit in ob.citations:
                    print(f"      -> {cit.doc_id} | {cit.section}")
    
    # State obligations
    for state_code, so in result.states.items():
        if state_code == "FED":
            continue
        print(f"\n{'=' * 40}")
        print(f"STATE: {state_code} (confidence: {so.confidence})")
        print(f"{'=' * 40}")
        
        if not so.obligations:
            print("  No requirements found in this corpus")
            if so.not_found_explanation:
                print(f"  Explanation: {so.not_found_explanation}")
            continue
        
        for i, ob in enumerate(so.obligations, 1):
            print(f"\n  [{i}] {ob.obligation}")
            if ob.trigger:
                print(f"      Trigger: {ob.trigger}")
            if ob.deadline:
                print(f"      Deadline: {ob.deadline}")
            if ob.notify_who:
                print(f"      Notify: {ob.notify_who}")
            if ob.required_content:
                print(f"      Content: {ob.required_content}")
            if ob.citations:
                for cit in ob.citations:
                    print(f"      -> {cit.doc_id} | {cit.section}")
                    if cit.quote:
                        quote_preview = cit.quote[:80] + "..." if len(cit.quote) > 80 else cit.quote
                        print(f"        \"{quote_preview}\"")
            if ob.evidence_artifacts:
                print(f"      Evidence: {', '.join(ob.evidence_artifacts)}")
            if ob.process_gates:
                print(f"      Gates: {', '.join(ob.process_gates)}")
    
    # Diffs
    if result.diffs:
        print(f"\n{'=' * 40}")
        print("KEY DIFFERENCES")
        print(f"{'=' * 40}")
        for diff in result.diffs:
            print(f"  {diff.field}: {diff.values}")
    
    # Optional: save full JSON
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("output") / f"rlm_result_{activity.replace(' ', '_')[:30]}.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "result": result.model_dump(),
                "cost": cost.to_dict() if cost else None,
            },
            f,
            indent=2,
        )
    if cost:
        print(
            f"\nCOST: {cost.llm_calls} LLM calls, {cost.total_tokens} tokens, "
            f"${cost.to_dict().get('estimated_cost_usd', 0.0):.4f} est."
        )
    print(f"[Saved full JSON to {output_path}]")


if __name__ == "__main__":
    main()
