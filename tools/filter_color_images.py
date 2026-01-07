"""Copy only color images from a folder into another folder.

This is intentionally non-destructive: it never deletes from the source.

An image is considered "color" if it has 3/4 channels and at least some pixel
differs across channels (i.e., not a grayscale image saved as RGB).

Example:
    python tools/filter_color_images.py --src assets/movie --dst assets/movie_color
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Copy only true-color images.")
    p.add_argument("--src", type=Path, required=True, help="Source folder of images.")
    p.add_argument("--dst", type=Path, required=True, help="Destination folder for color images.")
    p.add_argument(
        "--exts",
        type=str,
        default=".jpg,.jpeg,.png,.bmp,.webp",
        help="Comma-separated extensions to consider (default: common image types).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the destination.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel workers (0=disable; default: 0).",
    )
    return p.parse_args()


def _is_true_color(image_path: Path) -> bool:
    import cv2
    import numpy as np

    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False

    if img.ndim == 2:
        return False

    if img.ndim != 3 or img.shape[2] not in (3, 4):
        return False

    rgb = img[:, :, :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # Fast rejection: if there exists any pixel where channels differ, it's color.
    # This treats "grayscale stored as RGB" as not color.
    b, g, r = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return bool(((b != g) | (g != r)).any())


def _iter_images(src: Path, exts: set[str]):
    for p in sorted(src.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() in exts:
            yield p


def _copy_one(src_path: Path, dst_dir: Path, overwrite: bool) -> tuple[str, bool, str | None]:
    try:
        is_color = _is_true_color(src_path)
    except Exception as exc:
        return (src_path.name, False, f"read_failed: {exc}")

    if not is_color:
        return (src_path.name, False, None)

    dst_path = dst_dir / src_path.name
    if dst_path.exists() and not overwrite:
        return (src_path.name, True, "exists")

    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return (src_path.name, True, "copied")


def main() -> None:
    args = _parse_args()
    src: Path = args.src
    dst: Path = args.dst
    exts = {e.strip().lower() for e in args.exts.split(",") if e.strip()}

    if not src.exists() or not src.is_dir():
        raise SystemExit(f"Source directory not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    images = list(_iter_images(src, exts))
    if not images:
        raise SystemExit(f"No images found under {src} with extensions {sorted(exts)}")

    workers = int(args.workers or 0)
    workers = min(workers, max(os.cpu_count() or 1, 1))

    kept = 0
    skipped = 0
    errors = 0

    if workers > 0:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_copy_one, p, dst, args.overwrite) for p in images]
            for fut in as_completed(futs):
                name, is_color, status = fut.result()
                if status and status.startswith("read_failed"):
                    errors += 1
                elif is_color:
                    kept += 1
                else:
                    skipped += 1
    else:
        for p in images:
            _, is_color, status = _copy_one(p, dst, args.overwrite)
            if status and status.startswith("read_failed"):
                errors += 1
            elif is_color:
                kept += 1
            else:
                skipped += 1

    print(f"Scanned: {len(images)}")
    print(f"Color kept: {kept}")
    print(f"Non-color skipped: {skipped}")
    print(f"Errors: {errors}")
    print(f"Saved to: {dst}")


if __name__ == "__main__":
    main()
