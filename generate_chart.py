"""Generate a full results visualization chart for the README.

This script:
- Auto-selects the latest 40-query eval JSONs for each (model, strategy)
- Plots accuracy + cost + runtime metrics side-by-side
- Writes docs/results_chart.png

Notes on pricing:
- We compute cost from token counts using PRICING_USD_PER_1M for transparency.
- If a model isn't in the pricing map, we fall back to the JSON's stored estimate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


EVAL_DIRS = [Path("eval/results"), Path("eval/_archive")]
OUTPUT_PATH = Path("docs/results_chart.png")

# Update these if you want different models in the README.
MODELS = ["gpt-4o-mini", "gpt-5.2"]

# Transparent cost assumptions (USD per 1M tokens)
PRICING_USD_PER_1M: Dict[str, Dict[str, float]] = {
    # Values based on the numbers used in this repo's write-up.
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-5.2": {"prompt": 1.75, "completion": 14.00},
}


@dataclass(frozen=True)
class RunMetrics:
    model: str
    strategy: str  # "rag" or "rlm"

    critical_miss_rate_pct: float
    negative_test_accuracy_pct: float
    false_obligation_rate_pct: float
    deadline_accuracy_pct: float
    notify_who_accuracy_pct: float

    llm_calls: int
    total_time_min: float
    estimated_cost_usd: float


def _safe_get(dct: Dict[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = dct
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_full_suite(data: Dict[str, Any]) -> bool:
    total = _safe_get(data, "summary", "total_queries", default=None)
    return total == 40


def _extract_model_and_strategy(data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    model = _safe_get(data, "experiment", "model", default=None)
    strategy = _safe_get(data, "experiment", "experiment", default=None)
    if isinstance(model, str) and isinstance(strategy, str):
        return model, strategy
    return None, None


def _find_latest_run(model: str, strategy: str) -> Tuple[Path, Dict[str, Any]]:
    candidates: Iterable[Path] = []
    files: list[Path] = []
    for d in EVAL_DIRS:
        if d.exists():
            files.extend(sorted(d.glob("eval_*.json"), key=lambda p: p.stat().st_mtime, reverse=True))

    for path in files:
        try:
            data = _load_json(path)
        except Exception:
            continue
        if not _is_full_suite(data):
            continue
        m, s = _extract_model_and_strategy(data)
        if m == model and s == strategy:
            return path, data

    raise FileNotFoundError(f"No 40-query eval JSON found for model={model!r} strategy={strategy!r}")


def _estimate_cost_from_tokens(model: str, prompt_tokens: int, completion_tokens: int, fallback_usd: float) -> float:
    pricing = PRICING_USD_PER_1M.get(model)
    if not pricing:
        return float(fallback_usd)
    return (prompt_tokens * pricing["prompt"] + completion_tokens * pricing["completion"]) / 1_000_000.0


def _extract_metrics(data: Dict[str, Any]) -> RunMetrics:
    model = _safe_get(data, "experiment", "model")
    strategy = _safe_get(data, "experiment", "experiment")
    if not isinstance(model, str) or not isinstance(strategy, str):
        raise ValueError("Missing experiment metadata (model/experiment) in eval JSON")

    summary = _safe_get(data, "summary", default={})
    cost = _safe_get(data, "experiment", "cost", default={})

    llm_calls = int(_safe_get(cost, "llm_calls", default=0) or 0)
    prompt_tokens = int(_safe_get(cost, "prompt_tokens", default=0) or 0)
    completion_tokens = int(_safe_get(cost, "completion_tokens", default=0) or 0)
    stored_estimated_cost = float(_safe_get(cost, "estimated_cost_usd", default=0.0) or 0.0)

    total_elapsed_seconds = float(_safe_get(summary, "total_elapsed_seconds", default=0.0) or 0.0)
    total_time_min = total_elapsed_seconds / 60.0

    estimated_cost_usd = _estimate_cost_from_tokens(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        fallback_usd=stored_estimated_cost,
    )

    def pct(x: Any) -> float:
        return float(x) * 100.0

    return RunMetrics(
        model=model,
        strategy=strategy,
        critical_miss_rate_pct=pct(_safe_get(summary, "critical_miss_rate", default=0.0) or 0.0),
        negative_test_accuracy_pct=pct(_safe_get(summary, "negative_test_accuracy", default=0.0) or 0.0),
        false_obligation_rate_pct=pct(_safe_get(summary, "false_obligation_rate", default=0.0) or 0.0),
        deadline_accuracy_pct=pct(_safe_get(summary, "deadline_accuracy", default=0.0) or 0.0),
        notify_who_accuracy_pct=pct(_safe_get(summary, "notify_who_accuracy", default=0.0) or 0.0),
        llm_calls=llm_calls,
        total_time_min=total_time_min,
        estimated_cost_usd=estimated_cost_usd,
    )


def _plot_metric(
    ax: plt.Axes,
    title: str,
    y_label: str,
    model_names: list[str],
    rag_vals: list[float],
    rlm_vals: list[float],
    higher_is_better: bool,
    value_formatter,
) -> None:
    x = np.arange(len(model_names))
    width = 0.32

    rag_color = "#2b6cb0"  # blue
    rlm_color = "#2f855a"  # green

    bars_rag = ax.bar(x - width / 2, rag_vals, width, label="RAG", color=rag_color)
    bars_rlm = ax.bar(x + width / 2, rlm_vals, width, label="RLM", color=rlm_color)

    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5)

    # find the best bar
    all_bars = list(bars_rag) + list(bars_rlm)
    all_vals = [float(b.get_height()) for b in all_bars]
    best_idx = int(np.argmax(all_vals) if higher_is_better else np.argmin(all_vals))
    best_bar = all_bars[best_idx]

    # Expand y-axis to leave room for labels and star
    y_max = max(all_vals) * 1.25
    ax.set_ylim(0, y_max)

    # Annotate values on bars
    for i, bar in enumerate(all_bars):
        height = float(bar.get_height())
        ax.annotate(
            value_formatter(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Mark best bar with a gold star using matplotlib marker (no Unicode issues)
    ax.plot(
        best_bar.get_x() + best_bar.get_width() / 2,
        float(best_bar.get_height()) + y_max * 0.08,
        marker="*",
        markersize=14,
        color="#d69e2e",
        markeredgecolor="#b7791f",
        markeredgewidth=0.5,
    )


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    # Load the latest runs for each model & strategy
    runs: Dict[Tuple[str, str], RunMetrics] = {}
    sources: Dict[Tuple[str, str], Path] = {}
    for model in MODELS:
        for strategy in ("rag", "rlm"):
            path, data = _find_latest_run(model=model, strategy=strategy)
            runs[(model, strategy)] = _extract_metrics(data)
            sources[(model, strategy)] = path

    model_labels = MODELS

    # Assemble series
    def series(getter):
        rag = [getter(runs[(m, "rag")]) for m in model_labels]
        rlm = [getter(runs[(m, "rlm")]) for m in model_labels]
        return rag, rlm

    miss_rag, miss_rlm = series(lambda r: r.critical_miss_rate_pct)
    neg_rag, neg_rlm = series(lambda r: r.negative_test_accuracy_pct)
    false_rag, false_rlm = series(lambda r: r.false_obligation_rate_pct)
    dead_rag, dead_rlm = series(lambda r: r.deadline_accuracy_pct)
    notify_rag, notify_rlm = series(lambda r: r.notify_who_accuracy_pct)
    cost_rag, cost_rlm = series(lambda r: r.estimated_cost_usd)
    time_rag, time_rlm = series(lambda r: r.total_time_min)
    calls_rag, calls_rlm = series(lambda r: float(r.llm_calls))

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    fig.suptitle(
        "RAG vs Controller-Driven (RLM) — Accuracy, Cost, and Runtime (40-query suite)",
        fontsize=14,
        fontweight="bold",
    )
    # Manual spacing to avoid overlap
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.05, right=0.98, hspace=0.35, wspace=0.30)

    _plot_metric(
        axes[0, 0],
        "Critical Miss Rate ↓",
        "%",
        model_labels,
        miss_rag,
        miss_rlm,
        higher_is_better=False,
        value_formatter=lambda v: f"{float(v):.1f}%",
    )

    _plot_metric(
        axes[0, 1],
        "Negative Test Accuracy ↑",
        "%",
        model_labels,
        neg_rag,
        neg_rlm,
        higher_is_better=True,
        value_formatter=lambda v: f"{float(v):.1f}%",
    )

    _plot_metric(
        axes[0, 2],
        "False Obligation Rate ↓",
        "%",
        model_labels,
        false_rag,
        false_rlm,
        higher_is_better=False,
        value_formatter=lambda v: f"{float(v):.1f}%",
    )

    _plot_metric(
        axes[0, 3],
        "Deadline Accuracy ↑",
        "%",
        model_labels,
        dead_rag,
        dead_rlm,
        higher_is_better=True,
        value_formatter=lambda v: f"{float(v):.1f}%",
    )

    _plot_metric(
        axes[1, 0],
        "Notify Who Accuracy ↑",
        "%",
        model_labels,
        notify_rag,
        notify_rlm,
        higher_is_better=True,
        value_formatter=lambda v: f"{float(v):.1f}%",
    )

    _plot_metric(
        axes[1, 1],
        "Estimated Cost (USD) ↓",
        "$",
        model_labels,
        cost_rag,
        cost_rlm,
        higher_is_better=False,
        value_formatter=lambda v: f"${float(v):.2f}",
    )

    _plot_metric(
        axes[1, 2],
        "Total Runtime (min) ↓",
        "min",
        model_labels,
        time_rag,
        time_rlm,
        higher_is_better=False,
        value_formatter=lambda v: f"{float(v):.1f}m",
    )

    _plot_metric(
        axes[1, 3],
        "LLM Calls ↓",
        "calls",
        model_labels,
        calls_rag,
        calls_rlm,
        higher_is_better=False,
        value_formatter=lambda v: f"{int(round(float(v)))}",
    )

    # One shared legend at bottom center
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=2,
        frameon=True,
        fontsize=11,
        fancybox=True,
        shadow=False,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")

    print(f"Chart saved to {OUTPUT_PATH}")
    print("Sources used:")
    for model in MODELS:
        for strategy in ("rag", "rlm"):
            print(f"  - {model:10s} {strategy:3s}: {sources[(model, strategy)]}")


if __name__ == "__main__":
    main()
