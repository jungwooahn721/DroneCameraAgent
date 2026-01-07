from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math

BBoxXYXY = Tuple[float, float, float, float]



# EVALUATION RESULT DATACLASS

@dataclass(frozen=True)
class EvaluationResult:
    name: str
    score: float
    details: Dict[str, Any]



# HELPERS

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


def _is_pathlike(value: Any) -> bool:
    import os

    return isinstance(value, (str, bytes, os.PathLike))


def _to_gray(image_bgr):
    import cv2

    if image_bgr is None:
        return None
    if _is_pathlike(image_bgr):
        return cv2.imread(str(image_bgr), cv2.IMREAD_GRAYSCALE)
    if not hasattr(image_bgr, "shape"):
        return None
    if len(image_bgr.shape) == 2:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def _to_pil_rgb(image_bgr):
    from PIL import Image

    if image_bgr is None:
        return None
    if _is_pathlike(image_bgr):
        try:
            return Image.open(image_bgr).convert("RGB")
        except Exception:
            return None
    if not hasattr(image_bgr, "shape"):
        return None
    if len(image_bgr.shape) == 2:
        try:
            return Image.fromarray(image_bgr).convert("RGB")
        except Exception:
            return None
    try:
        import cv2
    except Exception:
        return None
    try:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    except Exception:
        return None



# BASE EVALUATOR CLASS

class Evaluator:
    name: str = "evaluator"

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        raise NotImplementedError



# SUBJECT COMPOSITION EVALUATORS

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


class RuleOfThirdsEvaluator(Evaluator):
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



# LOW-LEVEL EVALUATORS

class BrightnessEvaluator(Evaluator):
    """Calculates the mean brightness of the image (0-255)."""

    name = "brightness"

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        import numpy as np

        gray = _to_gray(image_bgr)
        if gray is None:
            return EvaluationResult(self.name, 0.0, {"reason": "image_load_failed"})
        score = float(np.mean(gray))
        return EvaluationResult(self.name, score, {"mean": score})


class LaplacianEvaluator(Evaluator):
    """Calculates the variance of the Laplacian (edge detection)."""

    name = "laplacian"

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        import cv2

        gray = _to_gray(image_bgr)
        if gray is None:
            return EvaluationResult(self.name, 0.0, {"reason": "image_load_failed"})
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return EvaluationResult(self.name, score, {"variance": score})


class StddevEvaluator(Evaluator):
    """Calculates the standard deviation of pixel intensities."""

    name = "stddev"

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        import numpy as np

        gray = _to_gray(image_bgr)
        if gray is None:
            return EvaluationResult(self.name, 0.0, {"reason": "image_load_failed"})
        score = float(np.std(gray))
        return EvaluationResult(self.name, score, {"stddev": score})




# AI-BASED EVALUATORS

"""class Siglip2Evaluator(Evaluator):
    #TODO: 
    name = "siglip2"

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-384",
        positive_prompt: str = "a photo of a man",
        negative_prompts: Optional[Sequence[str]] = None,
    ) -> None:
        self.model_name = model_name
        self.positive_prompt = positive_prompt
        if negative_prompts is None:
            self.negative_prompts = [
                "a black image",
                "nothing",
                "solid color",
                "noise",
                "extreme close-up of texture",
                "blurry image",
            ]
        else:
            self.negative_prompts = list(negative_prompts)
        self._processor = None
        self._model = None
        self._device = None

    def _load_model(self) -> Optional[str]:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except Exception as exc:
            return str(exc)

        if self._model is not None:
            return None
        try:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name, torch_dtype=dtype)
            self._model.to(self._device)
            self._model.eval()
            return None
        except Exception as exc:
            return str(exc)"

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        try:
            import torch
        except Exception as exc:
            return EvaluationResult(self.name, 0.0, {"reason": "missing_dependency", "error": str(exc)})

        err = self._load_model()
        if err is not None:
            return EvaluationResult(self.name, 0.0, {"reason": "model_load_failed", "error": err})

        image = _to_pil_rgb(image_bgr)
        if image is None:
            return EvaluationResult(self.name, 0.0, {"reason": "image_load_failed"})

        texts = [self.positive_prompt] + self.negative_prompts
        inputs = self._processor(text=texts, images=image, padding=True, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            score = probs[0][0].item()

        return EvaluationResult(
            self.name,
            float(score),
            {"positive_prompt": self.positive_prompt, "negative_prompts": list(self.negative_prompts)},
        )"""


class QAlignEvaluator(Evaluator):
    name = "qalign"

    def __init__(self) -> None:
        self._model = None
        self._device = None
        self._load_error: Optional[str] = None

    def _load_model(self) -> Optional[str]:
        try:
            import torch
            import pyiqa
        except Exception as exc:
            return str(exc)

        if self._model is not None:
            return None

        # If we already tried and failed, don't spam load attempts.
        if self._load_error is not None:
            return self._load_error

        # Q-Align in pyiqa often pulls in transformers/accelerate/bitsandbytes.
        # On some systems, a broken CUDA/bitsandbytes setup causes model init to fail.
        # Prefer CUDA if available, but fall back to CPU automatically.
        devices_to_try = [torch.device("cuda"), torch.device("cpu")] if torch.cuda.is_available() else [torch.device("cpu")]

        last_err: Optional[str] = None
        for dev in devices_to_try:
            self._device = dev
            print(f"Loading Q-Align model on {self._device}...")
            try:
                self._model = pyiqa.create_metric("qalign", device=self._device)
                self._load_error = None
                return None
            except Exception as exc:
                last_err = str(exc)
                self._model = None
                continue

        self._load_error = last_err or "unknown_error"
        print(f"Failed to create Q-Align metric: {self._load_error}")
        return self._load_error

    def evaluate(self, image_bgr, detections: Optional[Iterable[Any]] = None) -> EvaluationResult:
        try:
            import torch
            from torchvision.transforms.functional import to_tensor
        except Exception as exc:
            return EvaluationResult(self.name, 0.0, {"reason": "missing_dependency", "error": str(exc)})

        err = self._load_model()
        if err is not None or self._model is None:
            return EvaluationResult(self.name, 0.0, {"reason": "model_load_failed", "error": err})

        image = _to_pil_rgb(image_bgr)
        if image is None:
            return EvaluationResult(self.name, 0.0, {"reason": "image_load_failed"})

        try:
            input_tensor = to_tensor(image).unsqueeze(0).to(self._device)
            with torch.no_grad():
                score = self._model(input_tensor, task_="quality")
            return EvaluationResult(self.name, float(score.item()), {})
        except Exception as exc:
            print(f"Error inferring Q-Align for input: {exc}")
            return EvaluationResult(self.name, 0.0, {"reason": "inference_error", "error": str(exc)})



# COMPOSITE EVALUATOR

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

def default_composition_evaluator() -> CompositeEvaluator:
    return CompositeEvaluator(
        evaluators=[
            ColorContrastEvaluator(),
            SubjectSizeEvaluator(),
            RuleOfThirdsEvaluator(),
            BreathingSpaceEvaluator(),
        ],
        weights=[1.0, 1.0, 1.0, 1.0],
        name="composition",
    )


if __name__ == "__main__":
    import cv2
    from src.detectors.detector import GroundingDINODetector

    IMAGE_PATH = "outputs/Koky_LuxuryHouse_0_deterministic_radius_4_pos_fixed_251231_102750/images/img_0003.png"
    CONFIG_PATH = "repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    WEIGHTS_PATH = "repos/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25
    CAPTION = "a man with black clothes"
    image = cv2.imread(IMAGE_PATH)
    model = GroundingDINODetector(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
    detections = model.detect_file(
        image_path=IMAGE_PATH,
        caption=CAPTION,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )       
    
    
    evaluator = default_composition_evaluator()
    result = evaluator.evaluate(image_bgr=image, detections=detections)
    print(f"Composition score: {result.score:.4f}")
    print("Details:", result.details)
    
    """
    Composition score: 0.6325
    Details: {'components': [
    {'name': 'color_contrast', 'score': 0.029035447147387018, 'details': {'subject_std': 68.63666150001279, 'background_std': 72.74164384158124, 'separation': 0.029035447147387018, 'det_score': 0.8605086207389832, 'area_frac': 0.19032145346260432}}, 
    {'name': 'subject_size', 'score': 0.9426585918744205, 'details': {'area_frac': 0.19032145346260432, 'target': 0.18, 'tolerance': 0.18, 'det_score': 0.8605086207389832}}, 
    {'name': 'rule_of_thirds', 'score': 0.5582473096267943, 'details': {'center': [248.23076629638672, 240.64374542236328], 'dmin_norm': 0.20403362265007988, 'softness': 0.35, 'det_score': 0.8605086207389832}}, 
    {'name': 'breathing_space', 'score': 1.0, 'details': {'margins': {'left': 0.36357465386390686, 'right': 0.39392322301864624, 'top': 0.07759538292884827, 'bottom': 0.1375807523727417}, 'min_margin': 0.07759538292884827, 'min_margin_frac': 0.05, 'target_margin_frac': 0.12, 'det_score': 0.8605086207389832}}
    ], 
    'weights': [1.0, 1.0, 1.0, 1.0]}
    """
    

    
# https://www.studiobinder.com/blog/ultimate-guide-to-camera-shots/#ELS

# Shot Types (by Size):
# Extreme Wide Shot (EWS)
# Long Shot (LS)
# Full Shot (FS)
# Medium Wide Shot (MWS)
# Cowboy Shot (CS)
# Medium Shot (MS)
# Medium Close-Up (MCU)
# Close-Up (CU)
# Extreme Close-Up (ECU)

# Shot Types (by Framing):

