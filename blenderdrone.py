import math
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class BlenderDrone:
    """
    No bpy. Image-only.
    - update(rgb, depth=None, collided=False) ingests observation.
    - next_command() outputs incremental turn/move.
    - apply_command(cmd) updates internal pose (and optional external callback).

    Localization:
      - DINOv2 patch tokens for scene & reference (transformers defaults)
      - heat(patch) = max cosine(scene_patch, ref_patch)
      - top-quantile threshold -> connected components -> boxes (no detector)
    """

    def __init__(
        self,
        reference_image_path,
        model_id="facebook/dinov2-small",
        device="cuda",
        step_size=0.35,
        yaw_step_deg=15.0,
        pitch_step_deg=0.0,
        scan_yaw_degs=(0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 180),
        top_frac=0.12,
        min_component_patches=3,
        max_boxes=5,
        detect_score_threshold=0.35,
        center_tol=0.15,
        use_depth=True,
        min_clearance_m=0.7,
        depth_far_m=10.0,
        loop_penalty=0.01,
        quantize_pos=0.5,
        verbose=False,
    ):
        self.device = device

        self.step_size = float(step_size)
        self.yaw_step = math.radians(float(yaw_step_deg))
        self.pitch_step = math.radians(float(pitch_step_deg))
        self.scan_yaw = [math.radians(float(d)) for d in scan_yaw_degs]

        self.top_frac = float(top_frac)
        self.min_component_patches = int(min_component_patches)
        self.max_boxes = int(max_boxes)

        self.detect_score_threshold = float(detect_score_threshold)
        self.center_tol = float(center_tol)

        self.use_depth = bool(use_depth)
        self.min_clearance_m = float(min_clearance_m)
        self.depth_far_m = float(depth_far_m)

        self.loop_penalty = float(loop_penalty)
        self.quantize_pos = float(quantize_pos)

        self.verbose = bool(verbose)

        # Pose (two rows)
        self.x, self.y, self.z = 0.0, 0.0, 0.0
        self.yaw, self.pitch, self.roll = 0.0, 0.0, 0.0

        # Optional external integration callbacks
        self.get_pose_cb = None
        self.set_pose_cb = None

        # Observation cache
        self.last_rgb = None
        self.last_depth = None
        self.last_collided = False

        # Memory
        self.visited = set()
        self.current_score = None
        self.prev_score = None
        self.best_score = -1e9

        # Behavior
        self.mode = "SCAN"
        self.scan_idx = 0

        # Transformers (defaults)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

        self.patch_size = int(getattr(self.model.config, "patch_size", 14))

        # Reference tokens (precompute)
        ref_pil = Image.open(reference_image_path).convert("RGB")
        self.ref_tokens = self._l2norm(self._patch_tokens(ref_pil)[0])

    # -----------------------
    # Integration (optional)
    # -----------------------
    def set_callbacks(self, get_pose, set_pose):
        self.get_pose_cb = get_pose
        self.set_pose_cb = set_pose

    def sync_from_external_pose(self):
        if self.get_pose_cb is None:
            return
        x, y, z, yaw, pitch, roll = self.get_pose_cb()
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.yaw, self.pitch, self.roll = float(yaw), float(pitch), float(roll)

    def push_pose_to_external(self):
        if self.set_pose_cb is None:
            return
        self.set_pose_cb(self.x, self.y, self.z, self.yaw, self.pitch, self.roll)

    # -----------------------
    # Image utils
    # -----------------------
    def _to_pil(self, rgb):
        if isinstance(rgb, Image.Image):
            return rgb.convert("RGB")
        if isinstance(rgb, np.ndarray):
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            return Image.fromarray(rgb, mode="RGB")
        raise TypeError("rgb must be PIL.Image or numpy uint8 HxWx3")

    def _l2norm(self, x, eps=1e-8):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    # -----------------------
    # DINO tokens / heatmap
    # -----------------------
    def _patch_tokens(self, rgb):
        pil = self._to_pil(rgb)
        inputs = self.processor(images=pil, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            out = self.model(pixel_values=pixel_values)
            toks = out.last_hidden_state[0, 1:, :]  # drop CLS => (N, D)

        _, _, Hm, Wm = pixel_values.shape
        gh = max(1, Hm // self.patch_size)
        gw = max(1, Wm // self.patch_size)

        # If token count mismatches derived grid, infer from token count
        n = toks.shape[0]
        if gh * gw != n:
            g = int(round(math.sqrt(n)))
            gh, gw = (g, g) if g * g == n else (g, max(1, n // g))

        return toks, (gh, gw), (Hm, Wm)

    def dino_heatmap(self, rgb):
        scene_toks, (gh, gw), _ = self._patch_tokens(rgb)
        scene_toks = self._l2norm(scene_toks)

        with torch.no_grad():
            sim = scene_toks @ self.ref_tokens.T
            best = sim.max(dim=1).values

        return best.reshape(gh, gw).float().cpu().numpy()

    # -----------------------
    # Heatmap -> boxes (normalized to model input)
    # -----------------------
    def _connected_components(self, mask):
        gh, gw = mask.shape
        seen = np.zeros_like(mask, dtype=np.uint8)
        comps = []

        for y in range(gh):
            for x in range(gw):
                if mask[y, x] == 0 or seen[y, x]:
                    continue
                q = [(y, x)]
                seen[y, x] = 1
                comp = []
                while q:
                    cy, cx = q.pop()
                    comp.append((cy, cx))
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < gh and 0 <= nx < gw and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = 1
                            q.append((ny, nx))
                comps.append(comp)
        return comps

    def _boxes_from_heat(self, heat, model_hw):
        gh, gw = heat.shape
        Hm, Wm = model_hw

        thr = float(np.quantile(heat.reshape(-1), 1.0 - self.top_frac))
        mask = (heat >= thr).astype(np.uint8)
        comps = self._connected_components(mask)
        if not comps:
            return []

        boxes = []
        for comp in comps:
            if len(comp) < self.min_component_patches:
                continue

            ys = [p[0] for p in comp]
            xs = [p[1] for p in comp]
            y1p, y2p = min(ys), max(ys) + 1
            x1p, x2p = min(xs), max(xs) + 1

            score = float(np.mean([heat[y, x] for (y, x) in comp]))

            # patch coords -> model pixels
            x1 = int(x1p * self.patch_size)
            y1 = int(y1p * self.patch_size)
            x2 = int(x2p * self.patch_size)
            y2 = int(y2p * self.patch_size)

            x1, x2 = max(0, min(Wm, x1)), max(0, min(Wm, x2))
            y1, y2 = max(0, min(Hm, y1)), max(0, min(Hm, y2))

            area = max(1, (x2 - x1) * (y2 - y1))
            area_norm = float(area) / float(max(1, Hm * Wm))

            # normalized box in [0,1] relative to model input
            box_norm = (x1 / Wm, y1 / Hm, x2 / Wm, y2 / Hm)

            boxes.append({"score": score, "box_norm": box_norm, "area_norm": area_norm})

        boxes.sort(key=lambda b: b["score"], reverse=True)
        return boxes[: self.max_boxes]

    def detect(self, rgb):
        heat = self.dino_heatmap(rgb)
        _, _, model_hw = self._patch_tokens(rgb)  # cheap-ish; keeps things simple & consistent
        boxes = self._boxes_from_heat(heat, model_hw)
        return boxes[0] if boxes else None

    # -----------------------
    # Depth safety (optional)
    # -----------------------
    def _forward_clearance(self, depth):
        d = np.array(depth, dtype=np.float32)
        if d.ndim != 2:
            raise ValueError("depth must be (H,W)")

        d = np.where(np.isfinite(d), d, np.nan)
        d = np.where(d > 0, d, np.nan)
        d = np.where(d < self.depth_far_m, d, np.nan)

        h, w = d.shape
        y0, y1 = int(h * 0.42), int(h * 0.58)
        x0, x1 = int(w * 0.42), int(w * 0.58)
        crop = d[y0:y1, x0:x1].reshape(-1)
        crop = crop[np.isfinite(crop)]
        if crop.size == 0:
            return 0.0
        return float(np.percentile(crop, 10.0))

    def is_forward_safe(self):
        if not self.use_depth:
            return True
        if self.last_depth is None:
            return True
        return self._forward_clearance(self.last_depth) >= self.min_clearance_m

    # -----------------------
    # Pose-only motion
    # -----------------------
    def _forward_vec(self):
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        fx, fy, fz = cp * cy, cp * sy, sp
        n = math.sqrt(fx * fx + fy * fy + fz * fz) + 1e-8
        return fx / n, fy / n, fz / n

    def apply_command(self, cmd):
        t = cmd.get("type")
        if t == "move":
            self.x += float(cmd.get("dx", 0.0))
            self.y += float(cmd.get("dy", 0.0))
            self.z += float(cmd.get("dz", 0.0))
        elif t == "turn":
            self.yaw += float(cmd.get("dyaw", 0.0))
            self.pitch += float(cmd.get("dpitch", 0.0))
            self.roll += float(cmd.get("droll", 0.0))
        else:
            raise ValueError("Unknown cmd type")
        self.push_pose_to_external()

    # -----------------------
    # Update + policy
    # -----------------------
    def _quant_key(self):
        q = self.quantize_pos
        return (round(self.x / q), round(self.y / q), round(self.z / q))

    def update(self, rgb, depth=None, collided=False):
        self.sync_from_external_pose()

        self.last_rgb = rgb
        self.last_depth = depth
        self.last_collided = bool(collided)

        key = self._quant_key()
        loop_pen = self.loop_penalty if key in self.visited else 0.0
        self.visited.add(key)

        det = self.detect(rgb)
        if det is None:
            score = -1.0 - loop_pen
        else:
            score = float(det["score"]) + 0.25 * float(math.log(1e-6 + det["area_norm"])) - loop_pen

        if self.current_score is None:
            self.current_score, self.prev_score = score, score
        else:
            self.prev_score, self.current_score = self.current_score, score

        self.best_score = max(self.best_score, score)

        if self.last_collided:
            self.mode = "SCAN"
            self.scan_idx = 0

        if self.verbose:
            print("mode:", self.mode, "score:", self.current_score, "det:", None if det is None else det["score"])

        return det

    def next_command(self):
        if self.last_rgb is None:
            return {"type": "turn", "dyaw": 0.0, "dpitch": 0.0, "droll": 0.0}

        det = self.detect(self.last_rgb)

        # If no detection -> scan
        if det is None or float(det["score"]) < self.detect_score_threshold:
            self.mode = "SCAN"
            dy = self.scan_yaw[self.scan_idx % len(self.scan_yaw)]
            self.scan_idx += 1
            return {"type": "turn", "dyaw": dy, "dpitch": 0.0, "droll": 0.0}

        self.mode = "TRACK"
        self.scan_idx = 0

        x1n, y1n, x2n, y2n = det["box_norm"]
        cx = 0.5 * (x1n + x2n)
        cy = 0.5 * (y1n + y2n)

        ex = (cx - 0.5) / (0.5 + 1e-8)
        ey = (cy - 0.5) / (0.5 + 1e-8)

        # Yaw to center
        if abs(ex) > self.center_tol:
            dyaw = float(np.clip(ex, -1.0, 1.0)) * self.yaw_step
            return {"type": "turn", "dyaw": dyaw, "dpitch": 0.0, "droll": 0.0}

        # Optional pitch to center
        if self.pitch_step > 0.0 and abs(ey) > self.center_tol:
            dpitch = float(np.clip(-ey, -1.0, 1.0)) * self.pitch_step
            return {"type": "turn", "dyaw": 0.0, "dpitch": dpitch, "droll": 0.0}

        # Move forward if safe
        if self.is_forward_safe():
            fx, fy, fz = self._forward_vec()
            return {"type": "move", "dx": fx * self.step_size, "dy": fy * self.step_size, "dz": fz * self.step_size}

        # Not safe: turn a bit and rescan
        self.mode = "SCAN"
        self.scan_idx += 1
        return {"type": "turn", "dyaw": self.yaw_step, "dpitch": 0.0, "droll": 0.0}
