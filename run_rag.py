#!/usr/bin/env python3
"""
run_rag.py - Run the single-pass RAG baseline for ad-hoc research.

This is the RAG counterpart to run_rlm.py.

Usage:
    python run_rag.py "incident reporting" TX
    python run_rag.py "cybersecurity incident reporting" TX MD GA
    python run_rag.py "incident reporting" FED
    python run_rag.py "incident reporting" TX --model gpt-4o-mini

Environment:
    OPENAI_API_KEY must be set (or in .env file)

Note:
    For the full, fair comparison experiment, use:
        python run_all_experiments.py
    or:
        python -m eval.run_eval --experiment rag
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
    pass

# Verify API key
if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not set")
    print("Set it via environment variable or create a .env file")
    sys.exit(1)

from eval.rag_runner import RAGRunner


def parse_args():
    parser = argparse.ArgumentParser(description="Run the RAG baseline for ad-hoc research")
    parser.add_argument("activity", type=str, help="Research activity, e.g. 'incident reporting'")
    parser.add_argument(
        "states",
        nargs="*",
        default=None,
        help="States to query (e.g. TX MD GA). Use ALL or omit for all states.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model name (default: gpt-4o)",
    )
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["hybrid", "bm25", "semantic"],
        help="Retrieval mode (default: hybrid)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of retrieved chunks to include as context (default: 10)",
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
    states = args.states

    if not states or (len(states) == 1 and states[0].upper() == "ALL"):
        states = None

    print(f"Activity: {activity}")
    print(f"States: {states if states else 'ALL'}")
    print(f"Retrieval mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"Top-k: {args.top_k}")
    print("-" * 60)

    runner = RAGRunner(
        model=args.model,
        temperature=0.0,
        retrieval_mode=args.mode,
        top_k=args.top_k,
    )

    # Default to all states in the gold standard corpus if not provided.
    # For ad-hoc usage, use the common state list used by the eval harness.
    if states is None:
        states = [
            "FED",
            "GA",
            "MD",
            "MI",
            "NH",
            "NJ",
            "OK",
            "PA",
            "TX",
            "VT",
        ]

    result, cost = runner.research(activity, states)

    # Optional: save full JSON
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("output") / f"rag_result_{activity.replace(' ', '_')[:30]}.json"
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

    print(f"[Saved full JSON to {output_path}]")
    if cost:
        print(
            f"COST: {cost.llm_calls} LLM calls, {cost.total_tokens} tokens, "
            f"${cost.to_dict().get('estimated_cost_usd', 0.0):.4f} est."
        )


if __name__ == "__main__":
    main()
