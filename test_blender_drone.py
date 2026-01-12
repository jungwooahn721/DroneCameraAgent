from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union
_REPO_ROOT = Path(__file__).resolve().parent
_SRC_ROOT = _REPO_ROOT / "src"

# Make repo modules importable when running inside Blender.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
from drones.blender_drone.blender_drone import BlenderDroneEnvironment
from drones.dronecore.drone import DroneVisionAgent, Pose
from src.scenes.scene import open_scene, set_nishita_sky
from utils import load_config

_DEFAULT_CONFIG = _REPO_ROOT / "configs/default_run_config.yaml"


def _abs_path(value: Union[str, Path]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (_REPO_ROOT / path).resolve()


def _log(message: str, log_path: Optional[Path]) -> None:
    print(message)
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def _pose_str(pose: Pose) -> str:
    position, orientation = pose
    return f"pos={position}, ori={orientation}"


def _pause(interactive: bool, prompt: str) -> None:
    if not interactive:
        return
    input(prompt)


def _resolve_target_prompt(
    cli_prompt: Optional[str], run_cfg: dict, drone_cfg: dict
) -> Optional[str]:
    if cli_prompt:
        prompt = str(cli_prompt).strip()
        return prompt if prompt else None
    for cfg in (drone_cfg, run_cfg):
        if not isinstance(cfg, dict):
            continue
        prompt = cfg.get("target_prompt")
        if prompt:
            prompt = str(prompt).strip()
            if prompt:
                return prompt
    return None


def _resolve_device(detector_cfg: dict, run_cfg: dict, log_path: Optional[Path]) -> str:
    device = detector_cfg.get("device") if isinstance(detector_cfg, dict) else None
    if device is None or str(device).strip() == "":
        device = run_cfg.get("device", "cuda") if isinstance(run_cfg, dict) else "cuda"
    device = str(device).strip() if device is not None else "cuda"
    if not device:
        device = "cuda"
    if device.lower().startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                _log("Device 'cuda' requested but torch has no CUDA; using CPU.", log_path)
                device = "cpu"
        except Exception as exc:
            _log(f"Device '{device}' requested but torch unavailable ({exc}); using CPU.", log_path)
            device = "cpu"
    return device


def _serialize_detection(detection: object) -> dict:
    if hasattr(detection, "as_dict"):
        return detection.as_dict()
    if isinstance(detection, dict):
        return detection
    box = getattr(detection, "box_xyxy", None) or getattr(detection, "bbox_xyxy", None) or (0, 0, 0, 0)
    label = getattr(detection, "label", "object")
    score = getattr(detection, "score", 0.0)
    return {
        "label": str(label),
        "score": float(score),
        "bbox_xyxy": [float(v) for v in box],
    }


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _run_detect_and_evaluate(
    drone: DroneVisionAgent,
    image_path: Optional[str],
    output_root: Path,
    log_path: Optional[Path],
    target_prompt: Optional[str],
) -> None:
    if image_path is None:
        _log("detect/evaluate: skipped (render path missing).", log_path)
        return
    if not target_prompt:
        _log(
            "detect/evaluate: skipped (target_prompt not set in config or --target-prompt).",
            log_path,
        )
        return

    detections = drone.detect_target(image_path, target_prompt=target_prompt)
    _log(f"detect_target: {len(detections)} detections for prompt={target_prompt!r}", log_path)

    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    det_json_path = analysis_dir / "detect_target.json"
    _write_json(
        {
            "image": str(image_path),
            "prompt": target_prompt,
            "detections": [_serialize_detection(d) for d in detections],
        },
        det_json_path,
    )
    _log(f"detect_target json: {det_json_path}", log_path)

    try:
        import cv2
        from detectors.detector import annotate_image

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            _log(f"detect_target viz skipped (failed to read image): {image_path}", log_path)
        else:
            annotated = annotate_image(image_bgr, detections)
            viz_path = analysis_dir / "detect_target_viz.png"
            cv2.imwrite(str(viz_path), annotated)
            _log(f"detect_target viz: {viz_path}", log_path)
    except Exception as exc:
        _log(f"detect_target viz skipped: {exc}", log_path)

    eval_result = drone.evaluate_image(image_path, detections)
    _log(f"evaluate_image: {eval_result.name} score={eval_result.score:.4f}", log_path)
    eval_json_path = analysis_dir / "evaluate_image.json"
    _write_json(
        {
            "image": str(image_path),
            "prompt": target_prompt,
            "name": eval_result.name,
            "score": float(eval_result.score),
            "details": eval_result.details,
        },
        eval_json_path,
    )
    _log(f"evaluate_image json: {eval_json_path}", log_path)


def _steps() -> Iterable[Tuple[str, Tuple[float, float, float], Tuple[float, float, float]]]:
    return [
        ("move_up", (0.0, 0.0, 0.5), (0.0, 0.0, 0.0)),
        ("move_forward", (0.5, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ("yaw_right", (0.0, 0.0, 0.0), (0.0, 0.0, 0.35)),
        ("move_left", (0.0, 0.5, 0.0), (0.0, 0.0, 0.0)),
    ]


def _script_args() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Quick Blender smoke test for DroneVisionAgent.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(_DEFAULT_CONFIG),
        help="Path to config (YAML/JSON).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to config run.output_dir/blender_test.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Pause after each step for manual inspection.",
    )
    parser.add_argument(
        "--target-prompt",
        type=str,
        default=None,
        help="Optional text prompt for detect_target/evaluate_image tests.",
    )
    return parser.parse_args(_script_args())


def main(
    config_path: str,
    output_dir: Optional[str],
    interactive: bool,
    target_prompt: Optional[str],
) -> None:
    config_path = _abs_path(config_path)
    config = load_config(config_path)

    run_cfg, scene_cfg, detector_cfg, drone_cfg = (
        config[key] for key in ("run", "scene", "detector", "drone")
    )

    output_root = _abs_path(output_dir) if output_dir else _abs_path(run_cfg["output_dir"]) / "blender_test"
    images_dir = output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "test_log.txt"

    scene_file = _abs_path(scene_cfg["scene_file"])
    if not scene_file.exists():
        raise FileNotFoundError(f"Missing Blender scene file: {scene_file}")
    camera_info = drone_cfg["camera"]
    camera_name = camera_info.get("name", "DroneCamera") if isinstance(camera_info, dict) else "DroneCamera"

    scene = open_scene(str(scene_file))
    set_nishita_sky(scene_cfg["sky_strength"])
    env = BlenderDroneEnvironment(
        scene=scene,
        camera_info=camera_info,
        camera_name=camera_name,
        output_dir=images_dir,
    )

    init_from_scene = bool(drone_cfg.get("init_from_scene", False))
    configured_position = tuple(drone_cfg.get("initial_position", (0, 0, 0)))
    configured_orientation = tuple(drone_cfg.get("initial_orientation", (0, 0, 0)))
    if init_from_scene:
        initial_position, initial_orientation = env.get_pose()
    else:
        initial_position = configured_position
        initial_orientation = configured_orientation
        env.move_to(initial_position, initial_orientation)

    detector_config = detector_cfg.get("config_path") or detector_cfg.get("config") or ""
    detector_weights = detector_cfg.get("weights_path") or detector_cfg.get("weights") or ""
    if not detector_config or not detector_weights:
        raise ValueError("Detector config/weights are required to initialize DroneVisionAgent.")

    detector_config_path = _abs_path(detector_config)
    detector_weights_path = _abs_path(detector_weights)
    if not detector_config_path.exists():
        raise FileNotFoundError(f"Missing detector config: {detector_config_path}")
    if not detector_weights_path.exists():
        raise FileNotFoundError(f"Missing detector weights: {detector_weights_path}")

    resolved_prompt = _resolve_target_prompt(target_prompt, run_cfg, drone_cfg)
    resolved_device = _resolve_device(detector_cfg, run_cfg, log_path)

    drone = DroneVisionAgent(
        name=drone_cfg.get("name", "DavianDrone"),
        camera_info=camera_info,
        initial_position=initial_position,
        initial_orientation=initial_orientation,
        device=resolved_device,
        detector_config_path=detector_config_path,
        detector_weights_path=detector_weights_path,
        target_prompt=resolved_prompt,
        target_image=None,
        env=env,
    )

    _log("=== Blender DroneVisionAgent Smoke Test ===", log_path)
    _log(f"Scene: {scene_file}", log_path)
    _log(f"Output: {images_dir}", log_path)
    _log(f"Detector device: {resolved_device}", log_path)
    if init_from_scene:
        _log(
            "init_from_scene=True: using camera pose from scene (config initial_position ignored).",
            log_path,
        )
        _log(
            f"Config initial pose: pos={configured_position}, ori={configured_orientation}",
            log_path,
        )
        _log("Set drone.init_from_scene=false to use config initial_position.", log_path)
    else:
        _log("init_from_scene=False: using config initial pose.", log_path)
    _log(f"Initial pose: {_pose_str(drone.get_pose())}", log_path)

    initial_render = drone.render()
    _log(f"Initial render: {initial_render}", log_path)
    _run_detect_and_evaluate(drone, initial_render, output_root, log_path, resolved_prompt)
    _pause(interactive, "Press Enter to start motion steps...")

    for name, delta_position, delta_orientation in _steps():
        drone.move(delta_position, delta_orientation)
        pose = drone.get_pose()
        render_path = drone.render()
        _log(f"{name}: {_pose_str(pose)} | render={render_path}", log_path)
        _pause(interactive, f"Press Enter to continue after '{name}'...")

    _log("Returning to initial pose...", log_path)
    drone.move_to(initial_position, initial_orientation)
    pose = drone.get_pose()
    render_path = drone.render()
    _log(f"return_home: {_pose_str(pose)} | render={render_path}", log_path)
    _pause(interactive, "Press Enter to finish.")


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.output_dir, args.interactive, args.target_prompt)


"""
CUDA_VISIBLE_DEVICES=7 blender -b -P test_blender_drone.py -- --config configs/default_run_config.yaml --target-prompt "a person"

blender -b -P test_blender_drone.py -- --config configs/default_run_config.yaml --interactive
CUDA_VISIBLE_DEVICES=0 /home/nas3_userM/minhopark/repos/repos4students/jungwooahn/RoboticCameraControl/blender-4.5.2-linux-x64/blender -b -P test_blender_drone.py -- --config configs/default_run_config.yaml
"""
