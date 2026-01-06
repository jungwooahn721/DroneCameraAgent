
import argparse
import json
import random
from datetime import datetime
from math import cos, pi, radians, sin, sqrt
from pathlib import Path

import bpy
from mathutils import Euler, Matrix, Vector

from scenes.scene import open_scene, set_nishita_sky


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Render random views of an object in a Blender scene.")
    parser.add_argument("--input_scene", default="assets/Koky_LuxuryHouse_0.blend")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--run_name")
    parser.add_argument("--sky_strength", type=float, default=1.0)
    parser.add_argument("--object_position", nargs=3, type=float)
    parser.add_argument("--hemisphere", action="store_true")
    parser.add_argument("--camera_radius_range", nargs=2, type=float, default=[2.0, 10.0])
    parser.add_argument("--camera_direction_offsets", nargs=3, type=float, default=[30.0, 30.0, 30.0], help="yaw pitch roll offsets in degrees")
    parser.add_argument("--num_images", type=int, default=200)
    parser.add_argument("--disable_gpu", action="store_true", help="force CPU rendering; by default tries OPTIX/CUDA")
    parser.add_argument("--seed", type=int, default=721, help="random seed for camera sampling")
    parser.add_argument("--focal_length", type=float)
    parser.add_argument("--sensor_width", type=float)
    parser.add_argument("--sensor_height", type=float)
    parser.add_argument("--resolution", nargs=2, type=int, default=[512, 512])
    split = argv.index("--") + 1 if "--" in argv else len(argv)
    return parser.parse_args(argv[split:])


def ensure_camera(scene):
    cam = scene.camera
    if cam is None:
        for obj in bpy.data.objects:
            if obj.type == "CAMERA":
                cam = obj
                break
    if cam is None:
        cam_data = bpy.data.cameras.new("RenderCamera")
        cam = bpy.data.objects.new("RenderCamera", cam_data)
        scene.collection.objects.link(cam)
    scene.camera = cam
    return cam


def set_sky(strength):
    set_nishita_sky(float(strength))


def set_render_settings(scene, resolution):
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"


def object_position_from_name(object_name):
    obj = bpy.data.objects.get(object_name)
    if obj is None:
        raise RuntimeError(f"Object '{object_name}' not found in the scene.")
    return obj.location.copy()


def random_direction(hemisphere):
    u = random.random()
    v = random.random()
    theta = 2 * pi * u
    z = v if hemisphere else 2 * v - 1
    r = sqrt(max(0.0, 1.0 - z * z))
    return Vector((r * cos(theta), r * sin(theta), z))


def look_at_matrix(cam_pos, target, world_up=Vector((0.0, 0.0, 1.0))):
    forward = (target - cam_pos).normalized()
    right = world_up.cross(forward)
    if right.length < 1e-6:
        right = Vector((0.0, 1.0, 0.0)).cross(forward)
    right.normalize()
    up = forward.cross(right).normalized()
    return Matrix(
        (
            (right.x, up.x, -forward.x),
            (right.y, up.y, -forward.y),
            (right.z, up.z, -forward.z),
        )
    )


def sample_pose(obj_pos, radius_range, offsets_deg, hemisphere):
    r_min, r_max = sorted(radius_range)
    direction = random_direction(hemisphere)
    radius = random.uniform(r_min, r_max)
    cam_pos = obj_pos + direction * radius
    base_rot = look_at_matrix(cam_pos, obj_pos)
    base_forward = (obj_pos - cam_pos).normalized()
    yaw_range, pitch_range, roll_range = offsets_deg
    yaw = random.uniform(-yaw_range, yaw_range)
    pitch = random.uniform(-pitch_range, pitch_range)
    roll = random.uniform(-roll_range, roll_range)
    offset = Euler((radians(roll), radians(pitch), radians(yaw)), "XYZ").to_matrix()
    rot_matrix = base_rot @ offset
    final_forward = (rot_matrix @ Vector((0.0, 0.0, -1.0))).normalized()
    final_up = (rot_matrix @ Vector((0.0, 1.0, 0.0))).normalized()
    return {
        "cam_pos": cam_pos,
        "radius": radius,
        "base_forward": base_forward,
        "base_up": (base_rot @ Vector((0.0, 1.0, 0.0))).normalized(),
        "rot_matrix": rot_matrix,
        "offsets": {"yaw": yaw, "pitch": pitch, "roll": roll},
        "final_forward": final_forward,
        "final_up": final_up,
    }


def vec(v):
    return [float(v.x), float(v.y), float(v.z)]


def configure_gpu(scene, disable_gpu=False):
    scene.render.engine = "CYCLES"
    if disable_gpu:
        scene.cycles.device = "CPU"
        return {"device": "CPU", "devices": []}
    cprefs_addon = bpy.context.preferences.addons.get("cycles")
    if cprefs_addon is None:
        scene.cycles.device = "CPU"
        return {"device": "CPU", "devices": []}
    cprefs = cprefs_addon.preferences
    chosen = {"device": "CPU", "devices": []}
    for dtype in ("OPTIX", "CUDA"):
        try:
            cprefs.compute_device_type = dtype
            cprefs.get_devices()
        except Exception:
            continue
        gpu_devices = [d for d in cprefs.devices if d.type == dtype]
        if not gpu_devices:
            continue
        for d in cprefs.devices:
            d.use = d.type == dtype
        scene.cycles.device = "GPU"
        chosen = {"device": dtype, "devices": [d.name for d in gpu_devices]}
        break
    else:
        cprefs.compute_device_type = "CPU"
        cprefs.get_devices()
        scene.cycles.device = "CPU"
    return chosen


def main(argv):
    args = parse_args(argv)
    random.seed(args.seed)
    blend_path = Path(args.input_scene)
    scene_stem = blend_path.stem
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    folder_name = f"{scene_stem}_{timestamp}"
    if args.run_name:
        folder_name = f"{scene_stem}_{args.run_name}_{timestamp}"
    run_dir = Path(args.output_dir) / folder_name
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    scene = open_scene(str(blend_path))
    camera = ensure_camera(scene)
    if args.focal_length is not None:
        camera.data.lens = args.focal_length
    if args.sensor_width is not None:
        camera.data.sensor_width = args.sensor_width
    if args.sensor_height is not None:
        camera.data.sensor_height = args.sensor_height
    if args.sky_strength > 0:
        set_sky(args.sky_strength)
    device_info = configure_gpu(scene, args.disable_gpu)
    set_render_settings(scene, args.resolution)

    object_name = scene_stem.split("_")[0]
    obj_pos = Vector(args.object_position) if args.object_position is not None else object_position_from_name(object_name)

    annotations = []
    for idx in range(args.num_images):
        pose = sample_pose(obj_pos, args.camera_radius_range, args.camera_direction_offsets, args.hemisphere)
        camera.matrix_world = Matrix.Translation(pose["cam_pos"]) @ pose["rot_matrix"].to_4x4()
        scene.render.filepath = str(images_dir / f"img_{idx:04d}.png")
        bpy.ops.render.render(write_still=True)
        annotations.append(
            {
                "image": f"images/img_{idx:04d}.png",
                "camera_position": vec(pose["cam_pos"]),
                "radius": pose["radius"],
                "object_position": vec(obj_pos),
                "base_forward": vec(pose["base_forward"]),
                "base_up": vec(pose["base_up"]),
                "offsets_deg": pose["offsets"],
                "final_forward": vec(pose["final_forward"]),
                "final_up": vec(pose["final_up"]),
            }
        )

    run_info = {
        "input_scene": str(blend_path),
        "output_dir": str(run_dir),
        "scene_stem": scene_stem,
        "object_name": object_name,
        "run_name": args.run_name,
        "created_at": timestamp,
        "options": {
            "sky_strength": args.sky_strength,
            "object_position": vec(obj_pos),
            "hemisphere": args.hemisphere,
            "camera_radius_range": args.camera_radius_range,
            "camera_direction_offsets": args.camera_direction_offsets,
            "num_images": args.num_images,
            "seed": args.seed,
            "focal_length": args.focal_length,
            "sensor_width": args.sensor_width,
            "sensor_height": args.sensor_height,
            "resolution": args.resolution,
            "render_device": device_info["device"],
            "render_devices": device_info["devices"],
        },
    }

    with (run_dir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    with (run_dir / "annotations.json").open("w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)

    print(f"Saved {len(annotations)} images to {run_dir}")


if __name__ == "__main__":
    import sys

    raw_argv = getattr(bpy.app, "argv", None)
    if raw_argv is None:
        raw_argv = sys.argv
    raw_argv = list(raw_argv)
    if "--" not in raw_argv and raw_argv and raw_argv[0].endswith(".py"):
        raw_argv = raw_argv[1:]
    main(raw_argv)

"""Example Usage:
CUDA_VISIBLE_DEVICES=5 blender -b -P render_object.py -- --input_scene assets/Koky_LuxuryHouse_0.blend --camera_direction_offsets 0 0 0 --camera_radius_range 8 8 --run_name deterministic_radius_8_pos_fixed --hemisphere --object_position -0.78 5.58 0.56
"""
