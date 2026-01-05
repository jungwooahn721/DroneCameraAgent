"""Filter images based on metric thresholds and visualize the winners.

Defaults match the ranges in memos/filter_out.txt:
    - brightness: 10 to 230
    - laplacian: >= 10
    - qalign: >= 2.5
    - siglip2: (ignored by default)
    - stddev: 20 to 90
Images failing any enabled bound are dropped. The remaining images are sorted by
qalign (desc) and visualized in a grid; a text file lists the kept images.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

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
DEFAULT_LIMIT = None  # show all by default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter images by metric ranges and visualize the kept ones."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Run directory containing annotations.json and images/.",
    )
    parser.add_argument("--brightness-min", type=float, default=10.0)
    parser.add_argument("--brightness-max", type=float, default=230.0)
    parser.add_argument("--laplacian-min", type=float, default=10.0)
    parser.add_argument("--laplacian-max", type=float)
    parser.add_argument("--qalign-min", type=float, default=2.5)
    parser.add_argument("--qalign-max", type=float)
    parser.add_argument("--siglip2-min", type=float)
    parser.add_argument("--siglip2-max", type=float)
    parser.add_argument("--stddev-min", type=float, default=20.0)
    parser.add_argument("--stddev-max", type=float, default=90.0)
    parser.add_argument(
        "--cols",
        type=int,
        default=DEFAULT_COLS,
        help=f"Number of columns in the grid (default: {DEFAULT_COLS}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Limit number of filtered images to visualize (after sorting by qalign).",
    )
    parser.add_argument(
        "--list-path",
        type=Path,
        help="Path to save filtered image list (default: <output_dir>/filtered_images.txt).",
    )
    parser.add_argument(
        "--grid-path",
        type=Path,
        help="Path to save filtered grid image (default: <output_dir>/viz/filtered_grid.png).",
    )
    return parser.parse_args()


def load_annotations(path: Path) -> List[dict]:
    if not path.exists():
        raise SystemExit(f"annotations.json not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def within_bounds(value: float, lo: float | None, hi: float | None) -> bool:
    if lo is not None and value < lo:
        return False
    if hi is not None and value > hi:
        return False
    return True


def build_thresholds(args: argparse.Namespace) -> Dict[str, tuple[float | None, float | None]]:
    return {
        "score_brightness": (args.brightness_min, args.brightness_max),
        "score_laplacian": (args.laplacian_min, args.laplacian_max),
        "score_qalign": (args.qalign_min, args.qalign_max),
        # siglip2 defaults to no filtering (None bounds) unless user sets them
        "score_siglip2": (args.siglip2_min, args.siglip2_max),
        "score_stddev": (args.stddev_min, args.stddev_max),
    }


def filter_items(
    annotations: Sequence[dict],
    thresholds: Dict[str, tuple[float | None, float | None]],
    run_dir: Path,
) -> List[Dict[str, object]]:
    kept = []
    for item in annotations:
        img_rel = item.get("image")
        if not img_rel:
            continue
        img_path = run_dir / img_rel
        ok = True
        for field, (lo, hi) in thresholds.items():
            # Skip metrics with no bounds
            if lo is None and hi is None:
                continue
            value = safe_float(item.get(field))
            if value is None or not within_bounds(value, lo, hi):
                ok = False
                break
        if ok:
            qalign = safe_float(item.get("score_qalign"))
            kept.append({"rel": img_rel, "path": img_path, "qalign": qalign})
    return kept


def sort_by_qalign(items: List[Dict[str, object]]) -> List[Dict[str, object]]:
    # Missing qalign values go to the end
    return sorted(items, key=lambda x: (x["qalign"] is None, -(x["qalign"] or -1e9)))


def write_list(items: Sequence[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in items:
            line = entry["rel"]
            if entry["qalign"] is not None:
                line += f"\tqalign={entry['qalign']:.4f}"
            f.write(line + "\n")
    print(f"Saved filtered image list to {path}")


def plot_grid(
    items: List[Dict[str, object]],
    grid_path: Path,
    cols: int,
):
    if cols <= 0:
        cols = DEFAULT_COLS
    total = len(items)
    rows = math.ceil(total / cols) if total > 0 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes).flatten()
    for ax in axes:
        ax.axis("off")

    for idx, entry in enumerate(items):
        ax = axes[idx]
        try:
            img = plt.imread(entry["path"])
        except Exception:
            img = np.full((64, 64, 3), 0.8, dtype=float)
        ax.imshow(img)
        qalign = entry["qalign"]
        title = f"#{idx+1}"
        if qalign is not None:
            title += f" | q={qalign:.3f}"
        ax.set_title(title, fontsize=9)
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

    fig.suptitle(f"Filtered images (sorted by qalign desc) — n={total}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(grid_path, dpi=220)
    plt.close(fig)
    print(f"Saved grid to {grid_path}")


def main():
    args = parse_args()
    annotations_path = args.output_dir / "annotations.json"
    annotations = load_annotations(annotations_path)
    thresholds = build_thresholds(args)

    kept = filter_items(annotations, thresholds, args.output_dir)
    if not kept:
        raise SystemExit("No images passed the filters.")

    kept = sort_by_qalign(kept)
    if args.limit is not None and args.limit > 0:
        kept = kept[: args.limit]

    list_path = args.list_path or (args.output_dir / "filtered_images.txt")
    write_list(kept, list_path)

    grid_path = args.grid_path or (args.output_dir / "viz" / "filtered_grid.png")
    plot_grid(kept, grid_path, args.cols)


if __name__ == "__main__":
    main()
