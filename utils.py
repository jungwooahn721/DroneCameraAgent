from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

PathLike = Union[str, Path]


def load_config(path: PathLike) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as f:
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except Exception as e:
                raise ImportError("PyYAML is required for YAML configs. Install `pyyaml`.") from e
            data = yaml.safe_load(f) or {}
        else:
            data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def resolve_path(
    path_value: Optional[Any],
    *,
    base_dir: Path,
    fallback: Optional[Path] = None,
    prefer_repo_root: bool = False,
    repo_root: Optional[Path] = None,
) -> Path:
    if path_value is not None and str(path_value).strip() != "":
        path_text = str(path_value)
        candidate = Path(path_text)
        if candidate.is_absolute():
            return candidate
        if path_text.startswith("./") or path_text.startswith("../"):
            return (base_dir / candidate).resolve()
        if repo_root is None:
            repo_root = base_dir
        repo_candidate = (repo_root / candidate).resolve()
        if prefer_repo_root:
            return repo_candidate
        base_candidate = (base_dir / candidate).resolve()
        if base_candidate.exists():
            return base_candidate
        if repo_candidate.exists():
            return repo_candidate
        return base_candidate
    if fallback is not None:
        return fallback
    raise ValueError("Missing required path in config.")


def to_tuple3(value: Any, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    raise ValueError(f"Expected 3 values, got: {value}")
