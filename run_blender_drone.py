from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from drones.blender_drone.blender_drone import BlenderDrone
from scenes.scene import open_scene, set_nishita_sky
from utils import load_config, resolve_path, to_tuple3

_DEFAULT_RUN_CONFIG = _REPO_ROOT / "configs/run/run_config.yaml"
_DEFAULT_SCENE_CONFIG = _REPO_ROOT / "configs/scene/default_scene_config.yaml"
_DEFAULT_DRONE_CONFIG = _REPO_ROOT / "configs/drone/default_drone_config.yaml"
_DEFAULT_DETECTOR_CONFIG = _REPO_ROOT / "repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
_DEFAULT_DETECTOR_WEIGHTS = _REPO_ROOT / "repos/GroundingDINO/weights/groundingdino_swint_ogc.pth"


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Run Blender Drone Simulation")
    parser.add_argument(
        "--run-config",
        type=str,
        default=str(_DEFAULT_RUN_CONFIG),
        help="Path to run config (YAML/JSON)",
    )
    args = parser.parse_args()
    return args


def main(run_config_path: str) -> None:
    # ----------- Stage 0: Read Scenes and Initialize Drone Camera -------------------
    run_config_path = resolve_path(run_config_path, base_dir=Path.cwd(), repo_root=_REPO_ROOT)
    run_cfg = load_config(run_config_path)

    scene_cfg_path = resolve_path(
        run_cfg.get("scene_config"),
        base_dir=run_config_path.parent,
        fallback=_DEFAULT_SCENE_CONFIG,
        repo_root=_REPO_ROOT,
    )
    drone_cfg_path = resolve_path(
        run_cfg.get("drone_config"),
        base_dir=run_config_path.parent,
        fallback=_DEFAULT_DRONE_CONFIG,
        repo_root=_REPO_ROOT,
    )

    scene_cfg = load_config(scene_cfg_path)
    drone_cfg = load_config(drone_cfg_path)

    output_dir = resolve_path(
        run_cfg.get("output_dir", "outputs"),
        base_dir=run_config_path.parent,
        prefer_repo_root=True,
        repo_root=_REPO_ROOT,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    stage = int(run_cfg.get("stage", 0))
    num_steps = int(run_cfg.get("num_steps", 100))
    device = str(run_cfg.get("device", "cuda"))
    target_prompt = run_cfg.get("target_prompt")
    if target_prompt is not None:
        target_prompt = str(target_prompt)

    detector_cfg = run_cfg.get("detector")
    if not isinstance(detector_cfg, dict):
        detector_cfg = {}
    detector_config_path = resolve_path(
        run_cfg.get("detector_config", detector_cfg.get("config", str(_DEFAULT_DETECTOR_CONFIG))),
        base_dir=run_config_path.parent,
        repo_root=_REPO_ROOT,
    )
    detector_weights_path = resolve_path(
        run_cfg.get("detector_weights", detector_cfg.get("weights", str(_DEFAULT_DETECTOR_WEIGHTS))),
        base_dir=run_config_path.parent,
        repo_root=_REPO_ROOT,
    )

    scene_file = resolve_path(scene_cfg.get("scene_file"), base_dir=scene_cfg_path.parent, repo_root=_REPO_ROOT)
    if not scene_file.exists():
        raise FileNotFoundError(f"Missing Blender scene file: {scene_file}")
    sky_strength = float(scene_cfg.get("sky_strength", 0.0))

    drone_name = str(drone_cfg.get("name", "drone"))
    initial_position = to_tuple3(drone_cfg.get("initial_position"), (0.0, 0.0, 0.0))
    initial_orientation = to_tuple3(drone_cfg.get("initial_orientation"), (0.0, 0.0, 0.0))
    init_from_scene = bool(drone_cfg.get("init_from_scene", True))

    camera_cfg = drone_cfg.get("camera") or {}
    if isinstance(camera_cfg, dict):
        camera_info: Any = dict(camera_cfg)
    else:
        camera_info = camera_cfg

    camera_name = str(drone_cfg.get("camera_name", "DroneCamera"))
    if isinstance(camera_info, dict):
        camera_name = str(camera_info.pop("name", camera_name))

    image_format_override = run_cfg.get("image_format")
    render_engine_override = run_cfg.get("render_engine")
    if isinstance(camera_info, dict):
        if image_format_override:
            camera_info["image_format"] = image_format_override
        if render_engine_override:
            camera_info["render_engine"] = render_engine_override

    scene = open_scene(str(scene_file))
    set_nishita_sky(sky_strength)

    # Initialize Drone with Camera and Vision Agent (Blender runtime).
    drone = BlenderDrone(
        name=drone_name,
        camera_info=camera_info,
        initial_position=initial_position,
        initial_orientation=initial_orientation,
        device=device,
        detector_config_path=str(detector_config_path),
        detector_weights_path=str(detector_weights_path),
        target_prompt=target_prompt,
        scene=scene,
        camera_name=camera_name,
        output_dir=str(images_dir),
        image_format=str(image_format_override) if image_format_override else None,
        render_engine=str(render_engine_override) if render_engine_override else None,
        init_from_scene=init_from_scene,
    )
    if stage <= 0:
        return

    # ----------- Stage 1: Search for Target Using Drone Camera -------------------
    # TODO: implement
    # target_detected = drone.search_for_target( ... )
    # drone.set_target(target_detected)

    # ----------- Stage 2: Evaluate Captured Images for Target Presence ------------
    # TODO: implement
    # best_viewpoints = drone.find_best_viewpoints(... metrics=["qalign, rule_of_thirds", ...])


if __name__ == "__main__":
    args = parse_args()
    main(args.run_config)
