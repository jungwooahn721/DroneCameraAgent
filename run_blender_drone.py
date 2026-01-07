from __future__ import annotations

from pathlib import Path

from src.drones.blender_drone.blender_drone import BlenderDrone
from src.scenes.scene import open_scene, set_nishita_sky
from utils import load_config

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _REPO_ROOT / "configs/default_run_config.yaml"

def _abs_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (_REPO_ROOT / path).resolve()


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Run Blender Drone Simulation")
    parser.add_argument(
        "--config",
        type=str,
        default=str(_DEFAULT_CONFIG),
        help="Path to config (YAML/JSON)",
    )
    args = parser.parse_args()
    return args


def main(config_path: str) -> None:
    # ----------- Stage 0: Read Scenes and Initialize Drone Camera -------------------
    config_path = _abs_path(config_path)
    config = load_config(config_path)

    run_cfg, scene_cfg, detector_cfg, drone_cfg = (
        config[key] for key in ("run", "scene", "detector", "drone")
    )

    images_dir = _abs_path(run_cfg["output_dir"]) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    scene_file = _abs_path(scene_cfg["scene_file"])
    if not scene_file.exists():
        raise FileNotFoundError(f"Missing Blender scene file: {scene_file}")
    sky_strength = scene_cfg["sky_strength"]

    camera_info = drone_cfg["camera"]
    camera_name = camera_info.get("name", "DroneCamera") if isinstance(camera_info, dict) else "DroneCamera"

    scene = open_scene(str(scene_file))
    set_nishita_sky(sky_strength)

    # Initialize Drone with Camera and Vision Agent (Blender runtime).
    drone = BlenderDrone(
        name=drone_cfg["name"],
        camera_info=camera_info,
        initial_position=tuple(drone_cfg["initial_position"]),
        initial_orientation=tuple(drone_cfg["initial_orientation"]),
        device=run_cfg["device"],
        detector_config_path=str(_abs_path(detector_cfg["config"])),
        detector_weights_path=str(_abs_path(detector_cfg["weights"])),
        target_prompt=run_cfg["target_prompt"],
        scene=scene,
        camera_name=camera_name,
        output_dir=str(images_dir),
        image_format=run_cfg["image_format"],
        render_engine=run_cfg["render_engine"],
        init_from_scene=drone_cfg["init_from_scene"],
    )
    

    # ----------- Stage 1: Search for Target Using Drone Camera -------------------
    # TODO
    # target_detected = drone.search_for_target( ... )
    # drone.set_target(target_detected)

    # ----------- Stage 2: Evaluate Captured Images for Target Presence ------------
    # TODO
    # best_viewpoints = drone.find_best_viewpoints(... metrics=["qalign, rule_of_thirds", ...])


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
