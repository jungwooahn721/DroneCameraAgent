"""Visualize ranked images for one or more metrics.

Given a render run folder (e.g., ``outputs/Koky_LuxuryHouse_0_251222_105729``)
and a metric name, this script draws a grid of images sorted by that metric.
Scores are overlaid for quick inspection. If no metric is provided, it creates
one canvas per score_* field in the annotations file.
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
DEFAULT_COLS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize ranked images for score_* metrics in annotations.json."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Run directory containing annotations.json and images/.",
    )
    parser.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        help="Metric name without the 'score_' prefix. Repeat for multiple metrics. "
        "If omitted, all score_* fields are visualized.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=DEFAULT_COLS,
        help=f"Number of columns in the grid (default: {DEFAULT_COLS}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of ranked images to visualize (after sorting). Default: all.",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort scores in ascending order (default: descending).",
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


def safe_float(value) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def load_ranked_items(
    annotations: Sequence[dict],
    metric_key: str,
    run_dir: Path,
    ascending: bool,
    limit: int | None,
) -> List[Dict[str, object]]:
    items = []
    for item in annotations:
        score = safe_float(item.get(metric_key))
        if score is None:
            continue
        img_rel = item.get("image")
        if not img_rel:
            continue
        img_path = run_dir / img_rel
        items.append({"score": score, "path": img_path, "rel": img_rel})

    if not items:
        return []

    items.sort(key=lambda x: x["score"], reverse=not ascending)
    if limit is not None and limit > 0:
        items = items[:limit]

    for idx, entry in enumerate(items):
        entry["rank"] = idx + 1
    return items


def plot_grid(
    items: List[Dict[str, object]],
    metric_label: str,
    viz_dir: Path,
    cols: int,
    ascending: bool,
) -> Path:
    viz_dir.mkdir(parents=True, exist_ok=True)
    if cols <= 0:
        cols = DEFAULT_COLS
    total = len(items)
    rows = math.ceil(total / cols)
    fig_width = 3 * cols
    fig_height = 3 * rows if rows > 0 else 3
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = np.atleast_2d(axes).flatten()

    for ax in axes:
        ax.axis("off")

    for idx, entry in enumerate(items):
        ax = axes[idx]
        path = Path(entry["path"])
        try:
            img = plt.imread(path)
        except Exception:
            img = np.full((64, 64, 3), 0.8, dtype=float)
        ax.imshow(img)
        score = entry["score"]
        rank = entry["rank"]
        ax.set_title(f"#{rank} | {score:.3f}", fontsize=9)
        ax.text(
            0.02,
            0.08,
            Path(entry["rel"]).name,
            fontsize=7,
            color="white",
            ha="left",
            va="bottom",
            transform=ax.transAxes,
            bbox={"facecolor": "black", "alpha": 0.35, "pad": 2, "edgecolor": "none"},
        )

    fig.suptitle(
        f"{metric_label} (sorted {'ascending' if ascending else 'descending'}) — {total} images",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = viz_dir / f"rankings_{metric_label}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    annotations_path = args.output_dir / "annotations.json"
    annotations = load_annotations(annotations_path)
    available = discover_score_keys(annotations)
    metrics = resolve_metrics(args.metrics, available)

    viz_dir = args.output_dir / "viz"
    for metric_key in metrics:
        metric_label = metric_key.replace(SCORE_PREFIX, "")
        ranked_items = load_ranked_items(
            annotations, metric_key, args.output_dir, args.ascending, args.limit
        )
        if not ranked_items:
            print(f"Skipping {metric_label}: no scores available.")
            continue
        grid_path = plot_grid(ranked_items, metric_label, viz_dir, args.cols, args.ascending)
        print(f"[{metric_label}] saved rankings grid to {grid_path}")


if __name__ == "__main__":
    main()
