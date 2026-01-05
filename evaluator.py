from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math

BBoxXYXY = Tuple[float, float, float, float]

@dataclass(frozen=True)
class EvaluationResult:
    name: str
    score: float
    details: Dict[str, Any]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _safe_div(n: float, d: float, default: float = 0.0) -> float:
    if d == 0:
        return default
    return n / d


def _box_area(box: BBoxXYXY) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_center(box: BBoxXYXY) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _pick_primary_box(
    boxes: Sequence[BBoxXYXY],
    scores: Optional[Sequence[float]] = None,
) -> Optional[Tuple[BBoxXYXY, Optional[float]]]:
    if not boxes:
        return None
    if scores is None or len(scores) != len(boxes):
        best_idx = max(range(len(boxes)), key=lambda i: _box_area(boxes[i]))
        return boxes[best_idx], None
    best_idx = max(range(len(boxes)), key=lambda i: float(scores[i]) * _box_area(boxes[i]))
    return boxes[best_idx], float(scores[best_idx])


def _as_boxes_and_scores(detections: Optional[Iterable[Any]]) -> Tuple[List[BBoxXYXY], List[float]]:
    """Accepts either:
    - list of dicts like {bbox_xyxy: [x1,y1,x2,y2], score: 0.9}
    - list of objects like detector.Detection with .box_xyxy and .score
    """
    if detections is None:
        return [], []
    boxes: List[BBoxXYXY] = []
    scores: List[float] = []
    for d in detections:
        if isinstance(d, dict):
            box = d.get("bbox_xyxy") or d.get("box_xyxy") or d.get("bbox")
            sc = d.get("score", 0.0)
        else:
            box = getattr(d, "bbox_xyxy", None) or getattr(d, "box_xyxy", None)
            sc = getattr(d, "score", 0.0)
        if box is None:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        boxes.append((x1, y1, x2, y2))
        scores.append(float(sc))
    return boxes, scores


class Evaluator:
    name: str = "evaluator"

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        raise NotImplementedError


class CompositeEvaluator(Evaluator):
    def __init__(
        self,
        evaluators: Sequence[Evaluator],
        weights: Optional[Sequence[float]] = None,
        name: str = "composite",
    ) -> None:
        self.name = name
        self.evaluators = list(evaluators)
        if weights is None:
            self.weights = [1.0] * len(self.evaluators)
        else:
            if len(weights) != len(self.evaluators):
                raise ValueError("weights must match evaluators length")
            self.weights = [float(w) for w in weights]

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        results = [e.evaluate(image_bgr, detections) for e in self.evaluators]
        wsum = sum(self.weights)
        score = 0.0
        if wsum > 0:
            score = sum(w * r.score for w, r in zip(self.weights, results)) / wsum
        details: Dict[str, Any] = {
            "components": [{"name": r.name, "score": r.score, "details": r.details} for r in results],
            "weights": self.weights,
        }
        return EvaluationResult(name=self.name, score=_clamp01(score), details=details)


class ColorContrastEvaluator(Evaluator):
    """Scores local contrast inside subject vs background.

    Heuristic:
    - Convert to grayscale
    - Compare stddev inside subject bbox vs outside
    - Higher difference => better separation
    """

    name = "color_contrast"

    def __init__(self, min_subject_area_frac: float = 0.01) -> None:
        self.min_subject_area_frac = float(min_subject_area_frac)

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        import numpy as np
        import cv2

        h, w = image_bgr.shape[:2]
        boxes, scores = _as_boxes_and_scores(detections)
        picked = _pick_primary_box(boxes, scores)
        if picked is None:
            return EvaluationResult(self.name, 0.0, {"reason": "no_detections"})
        box, det_score = picked
        area_frac = _safe_div(_box_area(box), float(w * h), 0.0)
        if area_frac < self.min_subject_area_frac:
            return EvaluationResult(self.name, 0.0, {"reason": "subject_too_small", "area_frac": area_frac})

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        mask[y1:y2, x1:x2] = 255

        subj = gray[mask == 255]
        bg = gray[mask == 0]
        if subj.size < 50 or bg.size < 50:
            return EvaluationResult(self.name, 0.0, {"reason": "insufficient_pixels"})

        subj_std = float(subj.std())
        bg_std = float(bg.std())
        # normalized separation: |subj-bg| / (subj+bg+eps)
        sep = abs(subj_std - bg_std) / (subj_std + bg_std + 1e-6)
        score = _clamp01(sep)
        return EvaluationResult(
            self.name,
            score,
            {
                "subject_std": subj_std,
                "background_std": bg_std,
                "separation": sep,
                "det_score": det_score,
                "area_frac": area_frac,
            },
        )


class SubjectSizeEvaluator(Evaluator):
    """Scores how 'good' the subject size is.
    Uses primary detection box area fraction. Prefers a mid-range subject size.
    """

    name = "subject_size"

    def __init__(self, target_area_frac: float = 0.18, tolerance: float = 0.18) -> None:
        self.target_area_frac = float(target_area_frac)
        self.tolerance = float(tolerance)

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        h, w = image_bgr.shape[:2]
        boxes, scores = _as_boxes_and_scores(detections)
        picked = _pick_primary_box(boxes, scores)
        if picked is None:
            return EvaluationResult(self.name, 0.0, {"reason": "no_detections"})
        box, det_score = picked
        area_frac = _safe_div(_box_area(box), float(w * h), 0.0)

        # Triangular score peaked at target_area_frac
        dist = abs(area_frac - self.target_area_frac)
        score = 1.0 - _safe_div(dist, self.tolerance, 1.0)
        score = _clamp01(score)
        return EvaluationResult(
            self.name,
            score,
            {"area_frac": area_frac, "target": self.target_area_frac, "tolerance": self.tolerance, "det_score": det_score},
        )


class RuleOfThridsEvaluator(Evaluator):
    """Scores how close the subject center is to rule-of-thirds intersections."""
    # or ... consider only the horizontal distance from vertical thirds lines? #TODO

    name = "rule_of_thirds"

    def __init__(self, softness: float = 0.35) -> None:
        # softness controls how quickly score decays with distance
        self.softness = float(softness)

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        h, w = image_bgr.shape[:2]
        boxes, scores = _as_boxes_and_scores(detections)
        picked = _pick_primary_box(boxes, scores)
        if picked is None:
            return EvaluationResult(self.name, 0.0, {"reason": "no_detections"})
        box, det_score = picked
        cx, cy = _box_center(box)

        # 4 intersection points
        pts = [
            (w / 3.0, h / 3.0),
            (2.0 * w / 3.0, h / 3.0),
            (w / 3.0, 2.0 * h / 3.0),
            (2.0 * w / 3.0, 2.0 * h / 3.0),
        ]
        dists = [math.hypot((cx - px) / w, (cy - py) / h) for px, py in pts]
        dmin = min(dists)
        # Map distance in normalized units to [0,1] with exponential falloff
        score = math.exp(-_safe_div(dmin, self.softness, 1.0))
        score = _clamp01(score)
        return EvaluationResult(
            self.name,
            score,
            {"center": [cx, cy], "dmin_norm": dmin, "softness": self.softness, "det_score": det_score},
        )


class BreathingSpaceEvaluator(Evaluator):
    """Scores how much margin exists around the subject box.
    Prefers some empty space to image edges (not too tight).
    """

    name = "breathing_space"

    def __init__(self, min_margin_frac: float = 0.05, target_margin_frac: float = 0.12) -> None:
        self.min_margin_frac = float(min_margin_frac)
        self.target_margin_frac = float(target_margin_frac)

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        h, w = image_bgr.shape[:2]
        boxes, scores = _as_boxes_and_scores(detections)
        picked = _pick_primary_box(boxes, scores)
        if picked is None:
            return EvaluationResult(self.name, 0.0, {"reason": "no_detections"})
        box, det_score = picked
        x1, y1, x2, y2 = box

        left = _safe_div(x1, w, 0.0)
        right = _safe_div(w - x2, w, 0.0)
        top = _safe_div(y1, h, 0.0)
        bottom = _safe_div(h - y2, h, 0.0)
        min_margin = min(left, right, top, bottom)

        if min_margin <= 0:
            return EvaluationResult(self.name, 0.0, {"reason": "touching_edge", "min_margin": min_margin})

        # score: 0 at min_margin_frac, 1 at target_margin_frac, then gently decays
        if min_margin < self.min_margin_frac:
            score = _safe_div(min_margin, self.min_margin_frac, 0.0)
        else:
            # decay after target
            if min_margin <= self.target_margin_frac:
                score = 1.0
            else:
                score = math.exp(-_safe_div(min_margin - self.target_margin_frac, self.target_margin_frac, 1.0))
        score = _clamp01(score)
        return EvaluationResult(
            self.name,
            score,
            {
                "margins": {"left": left, "right": right, "top": top, "bottom": bottom},
                "min_margin": min_margin,
                "min_margin_frac": self.min_margin_frac,
                "target_margin_frac": self.target_margin_frac,
                "det_score": det_score,
            },
        )


def default_composition_evaluator() -> CompositeEvaluator:
    return CompositeEvaluator(
        evaluators=[
            ColorContrastEvaluator(),
            SubjectSizeEvaluator(),
            RuleOfThridsEvaluator(),
            BreathingSpaceEvaluator(),
        ],
        weights=[1.0, 1.0, 1.0, 1.0],
        name="composition",
    )
