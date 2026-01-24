#!/usr/bin/env python3
"""
run_all_experiments.py
----------------------
Run all experiment configurations SEQUENTIALLY and generate a comparison report.

Usage:
    python run_all_experiments.py
    python run_all_experiments.py --model gpt-4o-mini  # Use different model
    python run_all_experiments.py --skip-run  # Just generate report from existing results

Note: Run with the venv Python: .\.venv\Scripts\python.exe run_all_experiments.py
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

EXPERIMENTS = ["rag", "rlm"]
RESULTS_DIR = Path("eval/results")

# Get the Python executable - prefer venv if running from workspace
def get_python_executable():
    """Get the correct Python executable, preferring the venv."""
    workspace = Path(__file__).parent
    venv_python = workspace / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    # Fallback to current interpreter
    return sys.executable


def run_experiments(model: str = None):
    """Run all experiments sequentially (RAG first, then RLM)."""
    print("=" * 70)
    print("RUNNING ALL EXPERIMENTS (SEQUENTIAL: RAG → RLM)")
    print("=" * 70)
    
    python_exe = get_python_executable()
    print(f"Using Python: {python_exe}")
    if model:
        print(f"Model: {model}")
    
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(EXPERIMENTS)}] Running experiment: {exp}")
        print("=" * 70)
        
        cmd = [python_exe, "-m", "eval.run_eval", "--experiment", exp]
        if model:
            cmd.extend(["--model", model])
        
        # Run with output going directly to terminal (not captured)
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
        )
        
        if result.returncode != 0:
            print(f"ERROR: Experiment {exp} failed with code {result.returncode}")
        else:
            print(f"[OK] Experiment {exp} completed")
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 70)


def find_latest_results():
    """Find the most recent result file for each experiment."""
    results = {}
    
    for exp in EXPERIMENTS:
        # Find all result files for this experiment
        pattern = f"eval_{exp}_*.json"
        files = list(RESULTS_DIR.glob(pattern))
        
        if files:
            # Get the most recent one
            latest = max(files, key=lambda f: f.stat().st_mtime)
            results[exp] = latest
        else:
            print(f"WARNING: No results found for {exp}")
    
    return results


def load_results(result_files: dict) -> dict:
    """Load all result JSON files."""
    data = {}
    for exp, filepath in result_files.items():
        with open(filepath) as f:
            data[exp] = json.load(f)
    return data


def generate_comparison_report(data: dict):
    """Generate detailed comparison report."""
    
    print("\n")
    print("=" * 90)
    print("EXPERIMENT COMPARISON REPORT")
    print("=" * 90)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # =========================================================================
    # TEST COVERAGE SUMMARY
    # =========================================================================
    print("=" * 90)
    print("TEST COVERAGE")
    print("=" * 90)
    print()
    
    first_exp = next(iter(data.values()))
    total_queries = first_exp["summary"]["total_queries"]
    positive_tests = first_exp["summary"]["queries_with_gold"]
    negative_tests = first_exp["summary"].get("negative_tests", 0)
    
    print(f"Total Test Cases: {total_queries}")
    print(f"  - Positive (obligation expected): {positive_tests}")
    print(f"  - Negative (no obligation expected): {negative_tests}")
    print()
    
    # =========================================================================
    # ACCURACY METRICS
    # =========================================================================
    # Get model name from data (all experiments should use same model)
    model_name = first_exp.get("experiment", {}).get("model", "unknown")
    print("=" * 90)
    print(f"ACCURACY METRICS ({model_name})")
    print("=" * 90)
    print()
    
    # Header
    header = f"{'Metric':<32}"
    for exp in EXPERIMENTS:
        short = exp.replace("rlm_", "").replace("_", "-")[:10]
        header += f" {short:>10}"
    print(header)
    print("-" * 90)
    
    # Critical Miss Rate
    row = f"{'Critical Miss Rate':<32}"
    for exp in EXPERIMENTS:
        if exp in data:
            rate = data[exp]["summary"]["critical_miss_rate"]
            row += f" {rate*100:>9.1f}%"
        else:
            row += f" {'N/A':>10}"
    print(row)
    
    # Negative Test Accuracy (new)
    row = f"{'Negative Test Accuracy':<32}"
    for exp in EXPERIMENTS:
        if exp in data:
            acc = data[exp]["summary"].get("negative_test_accuracy", 0)
            n = data[exp]["summary"].get("negative_tests", 0)
            if n > 0:
                row += f" {acc*100:>9.1f}%"
            else:
                row += f" {'N/A':>10}"
        else:
            row += f" {'N/A':>10}"
    print(row)
    
    # Citation Validity
    row = f"{'Citation Validity':<32}"
    for exp in EXPERIMENTS:
        if exp in data:
            rate = data[exp]["summary"]["citation_validity_rate"]
            row += f" {rate*100:>9.1f}%"
        else:
            row += f" {'N/A':>10}"
    print(row)
    
    # Deadline Accuracy
    row = f"{'Deadline Accuracy':<32}"
    for exp in EXPERIMENTS:
        if exp in data:
            rate = data[exp]["summary"]["deadline_accuracy"]
            row += f" {rate*100:>9.1f}%"
        else:
            row += f" {'N/A':>10}"
    print(row)
    
    # Notify Who Accuracy
    row = f"{'Notify Who Accuracy':<32}"
    for exp in EXPERIMENTS:
        if exp in data:
            rate = data[exp]["summary"]["notify_who_accuracy"]
            row += f" {rate*100:>9.1f}%"
        else:
            row += f" {'N/A':>10}"
    print(row)
    
    # False Obligation Rate
    row = f"{'False Obligation Rate':<32}"
    for exp in EXPERIMENTS:
        if exp in data:
            rate = data[exp]["summary"]["false_obligation_rate"]
            row += f" {rate*100:>9.1f}%"
        else:
            row += f" {'N/A':>10}"
    print(row)
    
    print()
    
    # Cost row
    header = f"{'Cost Metrics':<32}"
    for exp in EXPERIMENTS:
        short = exp.replace("rlm_", "").replace("_", "-")[:10]
        header += f" {short:>10}"
    print(header)
    print("-" * 90)
    
    # Estimated Cost
    row = f"{'Estimated Cost (USD)':<32}"
    for exp in EXPERIMENTS:
        if exp in data and "experiment" in data[exp] and "cost" in data[exp]["experiment"]:
            cost = data[exp]["experiment"]["cost"]["estimated_cost_usd"]
            row += f" ${cost:>8.4f}"
        else:
            row += f" {'N/A':>10}"
    print(row)
    
    # LLM Calls
    row = f"{'LLM Calls':<32}"
    for exp in EXPERIMENTS:
        if exp in data and "experiment" in data[exp] and "cost" in data[exp]["experiment"]:
            calls = data[exp]["experiment"]["cost"]["llm_calls"]
            row += f" {calls:>10}"
        else:
            row += f" {'N/A':>10}"
    print(row)
    
    # Total Time
    row = f"{'Total Time (s)':<32}"
    for exp in EXPERIMENTS:
        if exp in data:
            time = data[exp]["summary"]["total_elapsed_seconds"]
            row += f" {time:>10.1f}"
        else:
            row += f" {'N/A':>10}"
    print(row)
    
    print()
    
    # =========================================================================
    # RLM VERIFICATION & REPAIR EFFECTIVENESS (only if RLM verification was used)
    # =========================================================================
    has_verification = any(
        data[exp]["summary"].get("total_rlm_verified", 0) > 0
        for exp in EXPERIMENTS if exp in data
    )
    if has_verification:
        print("=" * 90)
        print("RLM VERIFICATION & REPAIR EFFECTIVENESS")
        print("=" * 90)
        print()
        
        header = f"{'Metric':<40}"
        for exp in EXPERIMENTS:
            short = exp.replace("rlm_", "").replace("_", "-")[:10]
            header += f" {short:>10}"
        print(header)
        print("-" * 90)
        
        # Obligations Verified
        row = f"{'Obligations Verified':<40}"
        for exp in EXPERIMENTS:
            if exp in data:
                verified = data[exp]["summary"].get("total_rlm_verified", 0)
                row += f" {verified:>10}"
            else:
                row += f" {'N/A':>10}"
        print(row)
        
        # Dropped by Verifier
        row = f"{'Dropped by Verifier (activity mismatch)':<40}"
        for exp in EXPERIMENTS:
            if exp in data:
                dropped = data[exp]["summary"].get("total_rlm_dropped", 0)
                row += f" {dropped:>10}"
            else:
                row += f" {'N/A':>10}"
        print(row)
        
        # Repair Attempts
        row = f"{'Repair Attempts':<40}"
        for exp in EXPERIMENTS:
            if exp in data:
                attempts = data[exp]["summary"].get("total_repair_attempts", 0)
                row += f" {attempts:>10}"
            else:
                row += f" {'N/A':>10}"
        print(row)
        
        # Deadline Repaired
        row = f"{'Deadline Repaired (accepted)':<40}"
        for exp in EXPERIMENTS:
            if exp in data:
                repaired = data[exp]["summary"].get("total_repaired_deadline", 0)
                row += f" {repaired:>10}"
            else:
                row += f" {'N/A':>10}"
        print(row)
        
        # Deadline Rejected
        row = f"{'Deadline Repair Rejected':<40}"
        for exp in EXPERIMENTS:
            if exp in data:
                rejected = data[exp]["summary"].get("total_rejected_deadline", 0)
                row += f" {rejected:>10}"
            else:
                row += f" {'N/A':>10}"
        print(row)
        
        # Notify Repaired
        row = f"{'Notify Who Repaired (accepted)':<40}"
        for exp in EXPERIMENTS:
            if exp in data:
                repaired = data[exp]["summary"].get("total_repaired_notify", 0)
                row += f" {repaired:>10}"
            else:
                row += f" {'N/A':>10}"
        print(row)
        
        # Notify Rejected
        row = f"{'Notify Who Repair Rejected':<40}"
        for exp in EXPERIMENTS:
            if exp in data:
                rejected = data[exp]["summary"].get("total_rejected_notify", 0)
                row += f" {rejected:>10}"
            else:
                row += f" {'N/A':>10}"
        print(row)
        
        print()
    
    # =========================================================================
    # PER-TASK COMPARISON
    # =========================================================================
    print("=" * 90)
    print("PER-TASK SUCCESS/FAILURE")
    print("=" * 90)
    print()
    print("Legend: Y = Found, X = Miss, D = Deadline, N = Notify, NEG = Negative Test")
    print("        For negative tests: Y = correctly found nothing, X = false positive")
    print()
    
    # Get all query IDs from the first experiment
    first_exp = next(iter(data.values()))
    queries = [(r["query_id"], r["state"], r["activity"], r.get("expected_no_obligation", False)) 
               for r in first_exp["results"]]
    
    # Header
    header = f"{'Query ID':<35} {'Type':<5}"
    for exp in EXPERIMENTS:
        short = exp.replace("rlm_", "").replace("_", "-")[:8]
        header += f" {short:^10}"
    print(header)
    print("-" * 95)
    
    for query_id, state, activity, is_negative in queries:
        test_type = "NEG" if is_negative else "POS"
        row = f"{query_id:<35} {test_type:<5}"
        
        for exp in EXPERIMENTS:
            if exp not in data:
                row += f" {'N/A':^10}"
                continue
            
            # Find this query in results
            result = None
            for r in data[exp]["results"]:
                if r["query_id"] == query_id:
                    result = r
                    break
            
            if result is None:
                row += f" {'N/A':^10}"
                continue
            
            # Build status string
            if is_negative or result.get("expected_no_obligation"):
                # Negative test
                if result.get("negative_test_passed"):
                    status = "Y"
                else:
                    status = "XFP"
            else:
                # Positive test
                if result["critical_miss"]:
                    status = "X"
                else:
                    status = "Y"
                    if result.get("deadline_match"):
                        status += "D"
                    if result.get("notify_who_match"):
                        status += "N"
            
            row += f" {status:^10}"
        
        print(row)
    
    print()
    
    # =========================================================================
    # ERROR ANALYSIS COMPARISON
    # =========================================================================
    print("=" * 90)
    print("ERROR CATEGORY BREAKDOWN")
    print("=" * 90)
    print()
    
    # Collect all error categories
    all_categories = set()
    for exp_data in data.values():
        if "error_analysis" in exp_data:
            all_categories.update(exp_data["error_analysis"].get("errors_by_category", {}).keys())
    
    header = f"{'Error Category':<30}"
    for exp in EXPERIMENTS:
        short = exp.replace("rlm_", "").replace("_", "-")[:10]
        header += f" {short:>10}"
    print(header)
    print("-" * 90)
    
    for category in sorted(all_categories):
        row = f"{category:<30}"
        for exp in EXPERIMENTS:
            if exp in data and "error_analysis" in data[exp]:
                count = data[exp]["error_analysis"].get("errors_by_category", {}).get(category, 0)
                row += f" {count:>10}"
            else:
                row += f" {'N/A':>10}"
        print(row)
    
    print()
    
    # =========================================================================
    # POSITIVE TEST RESULTS BY STATE
    # =========================================================================
    print("=" * 90)
    print("POSITIVE TESTS BY STATE (Miss Rate)")
    print("=" * 90)
    print("Note: Lower is better. Shows miss rate for positive tests only (where obligations exist).")
    print()
    
    # Get all states and separate positive vs negative tests
    # We need to look at individual results to categorize by state
    positive_states = {}
    negative_states = {}
    
    first_data = list(data.values())[0]
    for result in first_data["results"]:
        state = result["state"]
        is_neg = result.get("expected_no_obligation", False)
        if is_neg:
            if state not in negative_states:
                negative_states[state] = 0
            negative_states[state] += 1
        else:
            if state not in positive_states:
                positive_states[state] = 0
            positive_states[state] += 1
    
    if positive_states:
        header = f"{'State':<8} {'Tests':>6}"
        for exp in EXPERIMENTS:
            short = exp.replace("rlm_", "").replace("_", "-")[:8]
            header += f" {short:>10}"
        print(header)
        print("-" * 90)
        
        for state in sorted(positive_states.keys()):
            queries = positive_states[state]
            row = f"{state:<8} {queries:>6}"
            
            for exp in EXPERIMENTS:
                if exp not in data:
                    row += f" {'N/A':>10}"
                    continue
                
                # Count misses for positive tests in this state
                misses = 0
                for result in data[exp]["results"]:
                    if result["state"] == state and not result.get("expected_no_obligation", False):
                        if result.get("critical_miss", False):
                            misses += 1
                
                if queries > 0:
                    miss_rate = misses / queries * 100
                    row += f" {miss_rate:>9.0f}%"
                else:
                    row += f" {'N/A':>10}"
            
            print(row)
        
        print()
    
    # =========================================================================
    # NEGATIVE TEST RESULTS BY STATE  
    # =========================================================================
    if negative_states:
        print("=" * 90)
        print("NEGATIVE TESTS BY STATE (Accuracy)")
        print("=" * 90)
        print("Note: Higher is better. Shows accuracy for negative tests (where no obligation should be found).")
        print("      A false positive = incorrectly extracting an obligation from a non-notice document.")
        print()
        
        header = f"{'State':<8} {'Tests':>6}"
        for exp in EXPERIMENTS:
            short = exp.replace("rlm_", "").replace("_", "-")[:8]
            header += f" {short:>10}"
        print(header)
        print("-" * 90)
        
        for state in sorted(negative_states.keys()):
            queries = negative_states[state]
            row = f"{state:<8} {queries:>6}"
            
            for exp in EXPERIMENTS:
                if exp not in data:
                    row += f" {'N/A':>10}"
                    continue
                
                # Count correct negatives (passed) for this state
                correct = 0
                for result in data[exp]["results"]:
                    if result["state"] == state and result.get("expected_no_obligation", False):
                        if result.get("negative_test_passed", False):
                            correct += 1
                
                if queries > 0:
                    accuracy = correct / queries * 100
                    row += f" {accuracy:>9.0f}%"
                else:
                    row += f" {'N/A':>10}"
            
            print(row)
        
        print()
    
    # =========================================================================
    # ABLATION INSIGHTS
    # =========================================================================
    print("=" * 90)
    print("KEY FINDINGS")
    print("=" * 90)
    print()
    
    if "rag" in data and "rlm" in data:
        rag_miss = data["rag"]["summary"]["critical_miss_rate"]
        rlm_miss = data["rlm"]["summary"]["critical_miss_rate"]
        diff = (rag_miss - rlm_miss) * 100
        print(f"1. RAG vs RLM (iterative multi-doc search value):")
        print(f"   RAG miss rate: {rag_miss*100:.1f}%, RLM miss rate: {rlm_miss*100:.1f}%")
        print(f"   -> Iterative search {'reduces' if diff > 0 else 'increases'} miss rate by {abs(diff):.1f}pp")
        print()
    
    # Cost efficiency
    if "rag" in data and "rlm" in data:
        if "experiment" in data["rag"] and "experiment" in data["rlm"]:
            rag_cost = data["rag"]["experiment"]["cost"]["estimated_cost_usd"]
            rlm_cost = data["rlm"]["experiment"]["cost"]["estimated_cost_usd"]
            rag_miss = data["rag"]["summary"]["critical_miss_rate"]
            rlm_miss = data["rlm"]["summary"]["critical_miss_rate"]
            
            print(f"2. Cost-Accuracy Tradeoff:")
            print(f"   RAG: ${rag_cost:.4f}, {rag_miss*100:.1f}% miss")
            print(f"   RLM: ${rlm_cost:.4f}, {rlm_miss*100:.1f}% miss")
            cost_ratio = rlm_cost / rag_cost if rag_cost > 0 else float('inf')
            print(f"   -> RLM costs {cost_ratio:.1f}x more")
            if rag_miss > rlm_miss:
                miss_reduction = (rag_miss - rlm_miss) * 100
                cost_per_pp = (rlm_cost - rag_cost) / miss_reduction if miss_reduction > 0 else 0
                print(f"   -> Cost per 1pp miss reduction: ${cost_per_pp:.4f}")
            print(f"   RAG: ${rag_cost:.4f}, {rag_miss*100:.1f}% miss")
            print(f"   RLM: ${rlm_cost:.4f}, {rlm_miss*100:.1f}% miss")
            cost_ratio = rlm_cost / rag_cost if rag_cost > 0 else float('inf')
            print(f"   -> RLM costs {cost_ratio:.1f}x more")
            if rag_miss > rlm_miss:
                miss_reduction = (rag_miss - rlm_miss) * 100
                cost_per_pp = (rlm_cost - rag_cost) / miss_reduction if miss_reduction > 0 else 0
                print(f"   -> Cost per 1pp miss reduction: ${cost_per_pp:.4f}")
    
    print()
    print("=" * 90)
    print("END OF REPORT")
    print("=" * 90)


def save_report(data: dict, output_path: Path):
    """Save the comparison report to a file."""
    import io
    import sys
    
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    
    generate_comparison_report(data)
    
    report = buffer.getvalue()
    sys.stdout = old_stdout
    
    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    # Also print to console
    print(report)
    print(f"\n[Report saved to {output_path}]")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-run", action="store_true", help="Skip running experiments, just generate report")
    parser.add_argument("--model", type=str, default=None, help="Model to use (default: gpt-4o). Options: gpt-4o, gpt-4o-mini, etc.")
    args = parser.parse_args()
    
    if not args.skip_run:
        run_experiments(model=args.model)
    
    # Find and load results
    result_files = find_latest_results()
    
    if not result_files:
        print("ERROR: No result files found")
        sys.exit(1)
    
    print(f"\nLoading results from:")
    for exp, path in result_files.items():
        print(f"  {exp}: {path.name}")
    
    data = load_results(result_files)
    
    # Generate and save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = RESULTS_DIR / f"comparison_report_{timestamp}.txt"
    save_report(data, report_path)


if __name__ == "__main__":
    main()
