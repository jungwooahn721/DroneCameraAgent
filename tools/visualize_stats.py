"""Visualize metric distributions from a render run.

Given a run folder (e.g., ``outputs/Koky_LuxuryHouse_0_251222_105729``) this
script reads ``annotations.json``, computes summary statistics for image
metrics, and saves simple plots.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")  # headless friendly
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "matplotlib is required for visualization. Install it with `pip install matplotlib`."
    ) from exc

SCORE_PREFIX = "score_"
DEFAULT_BINS = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize distributions of metrics stored in annotations.json."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Path to the run directory (must contain annotations.json).",
    )
    parser.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        help="Metric name without the 'score_' prefix. Repeat for multiple metrics. "
        "If omitted, all score_* fields in annotations.json are used.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help=f"Number of bins for histograms (default: {DEFAULT_BINS}).",
    )
    return parser.parse_args()


def load_annotations(path: Path) -> List[dict]:
    if not path.exists():
        raise SystemExit(f"annotations.json not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_score_keys(annotations: Sequence[dict]) -> List[str]:
    keys = set()
    for item in annotations:
        for key in item.keys():
            if key.startswith(SCORE_PREFIX):
                keys.add(key)
    return sorted(keys)


def resolve_metrics(requested: Iterable[str] | None, available: Sequence[str]) -> List[str]:
    if not available:
        raise SystemExit("No score_* fields found in annotations.json. Run score_images.py first.")
    if not requested:
        return list(available)

    resolved = []
    available_set = set(available)
    for name in requested:
        key = name if name.startswith(SCORE_PREFIX) else f"{SCORE_PREFIX}{name}"
        if key not in available_set:
            readable = ", ".join(sorted(m.replace(SCORE_PREFIX, "") for m in available))
            raise SystemExit(f"Metric '{name}' not found. Available metrics: {readable}")
        resolved.append(key)
    return resolved


def collect_values(annotations: Sequence[dict], field: str) -> List[float]:
    values = []
    for item in annotations:
        raw = item.get(field)
        if raw is None or isinstance(raw, bool):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isnan(value) or math.isinf(value):
            continue
        values.append(value)
    return values


def compute_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    percentiles = np.percentile(arr, [5, 25, 50, 75, 95])
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p05": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "median": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p95": float(percentiles[4]),
    }


def plot_overview(metrics_data: List[Dict[str, object]], viz_dir: Path, bins: int) -> Path:
    """Draw all metrics in a single canvas (hist + box per row)."""
    if not metrics_data:
        raise SystemExit("No metrics to plot.")

    viz_dir.mkdir(parents=True, exist_ok=True)
    rows = len(metrics_data)
    bins = bins if bins and bins > 0 else DEFAULT_BINS
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3.2 * rows))
    axes = np.atleast_2d(axes)

    for idx, data in enumerate(metrics_data):
        label = str(data["label"])
        values = np.asarray(data["values"], dtype=float)
        summary = data["summary"]

        hist_ax, box_ax = axes[idx]
        hist_ax.hist(values, bins=bins, color="#4C78A8", edgecolor="white", alpha=0.85)
        hist_ax.axvline(summary["mean"], color="#E45756", linestyle="--", linewidth=1.3, label=f"mean {summary['mean']:.3f}")
        hist_ax.axvline(summary["median"], color="#54A24B", linestyle=":", linewidth=1.3, label=f"median {summary['median']:.3f}")
        hist_ax.set_title(f"{label} distribution (n={summary['count']})")
        hist_ax.set_xlabel(label)
        hist_ax.set_ylabel("Count")
        hist_ax.legend(fontsize="small")

        box = box_ax.boxplot(values, vert=True, patch_artist=True, labels=[label])
        for patch in box["boxes"]:
            patch.set_facecolor("#72B7B2")
            patch.set_alpha(0.9)
        box_ax.set_title(f"{label} boxplot")

    fig.tight_layout()
    overview_path = viz_dir / "overview.png"
    fig.savefig(overview_path, dpi=220)
    plt.close(fig)
    return overview_path


def main():
    args = parse_args()
    annotations_path = args.output_dir / "annotations.json"
    annotations = load_annotations(annotations_path)
    available = discover_score_keys(annotations)
    metrics = resolve_metrics(args.metrics, available)

    stats_out: Dict[str, Dict[str, float | str | int]] = {}
    viz_dir = args.output_dir / "viz"
    overview_data: List[Dict[str, object]] = []

    for metric_key in metrics:
        metric_label = metric_key.replace(SCORE_PREFIX, "")
        values = collect_values(annotations, metric_key)
        if not values:
            print(f"Skipping {metric_label}: no numeric values found.")
            continue
        summary = compute_stats(values)
        stats_out[metric_label] = {
            "field": metric_key,
            **summary,
        }
        overview_data.append({"label": metric_label, "values": values, "summary": summary})
        print(
            f"[{metric_label}] count={summary['count']} mean={summary['mean']:.3f} "
            f"std={summary['std']:.3f} min={summary['min']:.3f} "
            f"median={summary['median']:.3f} max={summary['max']:.3f}"
        )

    if not stats_out:
        raise SystemExit("No metrics processed. Ensure annotations.json contains score_* fields.")

    stats_path = args.output_dir / "stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats_out, f, indent=2)

    overview_path = plot_overview(overview_data, viz_dir, args.bins)
    print(f"Wrote stats for {len(stats_out)} metric(s) to {stats_path}")
    print(f"Combined canvas saved to {overview_path}")


if __name__ == "__main__":
    main()
