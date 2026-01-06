from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import bpy
from mathutils import Vector, Euler
from drones.dronecore.drone import DroneVisionAgent


def _info_value(info: Any, key: str, default: Any = None) -> Any:
    if info is None:
        return default
    if isinstance(info, dict):
        return info.get(key, default)
    return getattr(info, key, default)


class BlenderDrone(DroneVisionAgent):
    def __init__(
        self,
        name: str,
        camera_info: Any = None,
        initial_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        initial_orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        device: str = "cuda",
        detector_config_path: str = "repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        detector_weights_path: str = "repos/GroundingDINO/weights/groundingdino_swint_ogc.pth",
        target_prompt: Optional[str] = None,
        target_image: Optional[str] = None,
        target_image_embedding: Optional[Any] = None,
        scene_path: Optional[str] = None,
        scene=None,
        camera=None,
        camera_name: str = "DroneCamera",
        output_dir: str = "outputs",
        image_format: Optional[str] = None,
        render_engine: Optional[str] = None,
        init_from_scene: bool = True,
    ) -> None:

        if scene_path:
            bpy.ops.wm.open_mainfile(filepath=str(scene_path))
            scene = bpy.context.scene
        self.scene = scene or bpy.context.scene

        self.blender_camera = self._resolve_camera(camera_name)
        self.scene.camera = self.blender_camera

        super().__init__(
            name=name,
            camera_info=camera_info,
            initial_position=initial_position,
            initial_orientation=initial_orientation,
            device=device,
            detector_config_path=detector_config_path,
            detector_weights_path=detector_weights_path,
            target_prompt=target_prompt,
            target_image=target_image,
            target_image_embedding=target_image_embedding,
        )

        self.camera = self.blender_camera
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_format = image_format or _info_value(self.camera_info, "image_format", "PNG")
        self.render_engine = render_engine or _info_value(self.camera_info, "render_engine", "CYCLES")
        self._render_index = 0

        self.set_pose_callbacks(self._get_pose, self._set_pose)
        self.set_camera_callbacks(self._apply_camera_info)
        self.set_render_callbacks(self._render_to_path)

        self._apply_camera_info(self.camera_info)
        if init_from_scene:
            self.update_pose_from_env()
        else:
            self.apply_pose_to_env()

    def _resolve_camera(self, camera_name: str):
        cam_data = bpy.data.cameras.new(camera_name)
        cam = bpy.data.objects.new(camera_name, cam_data)
        self.scene.collection.objects.link(cam)
        return cam

    def _update_scene(self) -> None:
        self.scene.update_tag()
        bpy.context.view_layer.update()

    def _get_pose(self, drone_name=None):
        loc = self.camera.location
        rot = self.camera.rotation_euler
        return {
            "position": (float(loc.x), float(loc.y), float(loc.z)),
            "orientation": (float(rot.z), float(rot.y), float(rot.x)),
        }

    def _set_pose(self, drone_name, pose: dict) -> None:
        if pose is None:
            return
        position = pose.get("position")
        orientation = pose.get("orientation")
        if position is not None:
            self.camera.location = Vector((float(position[0]), float(position[1]), float(position[2])))
        if orientation is not None:
            roll, pitch, yaw = orientation
            self.camera.rotation_euler = Euler((float(roll), float(pitch), float(yaw)), "XYZ")
        self._update_scene()

    def _apply_camera_info(self, camera_info: Any = None) -> None:
        info = camera_info if camera_info is not None else self.camera_info
        if info is None:
            return

        lens = _info_value(info, "focal_length", None)
        if lens is None:
            lens = _info_value(info, "lens", None)
        if lens is not None:
            self.camera.data.lens = float(lens)

        sensor_width = _info_value(info, "sensor_width", None)
        if sensor_width is not None:
            self.camera.data.sensor_width = float(sensor_width)

        sensor_height = _info_value(info, "sensor_height", None)
        if sensor_height is not None:
            self.camera.data.sensor_height = float(sensor_height)

        clip_start = _info_value(info, "clip_start", None)
        if clip_start is not None:
            self.camera.data.clip_start = float(clip_start)

        clip_end = _info_value(info, "clip_end", None)
        if clip_end is not None:
            self.camera.data.clip_end = float(clip_end)

        resolution = _info_value(info, "resolution", None)
        if resolution is None:
            res_x = _info_value(info, "resolution_x", None)
            res_y = _info_value(info, "resolution_y", None)
            if res_x is not None and res_y is not None:
                resolution = (res_x, res_y)
        if resolution is not None:
            self.scene.render.resolution_x = int(resolution[0])
            self.scene.render.resolution_y = int(resolution[1])
            self.scene.render.resolution_percentage = 100

        engine = _info_value(info, "render_engine", self.render_engine)
        if engine:
            self.scene.render.engine = str(engine)
            self.render_engine = str(engine)

        image_format = _info_value(info, "image_format", self.image_format)
        if image_format:
            self.scene.render.image_settings.file_format = str(image_format)
            self.image_format = str(image_format)

        self._update_scene()

    def _image_extension(self) -> str:
        fmt = str(self.image_format or "PNG").lower()
        mapping = {
            "jpeg": "jpg",
            "jpg": "jpg",
            "png": "png",
            "tiff": "tiff",
            "tif": "tif",
            "open_exr": "exr",
        }
        return mapping.get(fmt, "png")

    def _next_output_path(self, output_path: Optional[str]) -> Path:
        if output_path:
            return Path(output_path)
        self._render_index += 1
        filename = f"frame_{self._render_index:06d}.{self._image_extension()}"
        return self.output_dir / filename

    def _render_to_path(self, output_path: Optional[str] = None) -> str:
        path = self._next_output_path(output_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        return str(path)

    def render(self, output_path: Optional[str] = None, return_image: bool = False):
        path = self._render_to_path(output_path)
        if not return_image:
            return path
        
        import cv2
        image = cv2.imread(str(path))
        return image, path

    def capture(
        self,
        output_path: Optional[str] = None,
        metrics=None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        self.update_pose_from_env()
        image_path = self._render_to_path(output_path)
        detections = self.detect_target(
            image_path,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        evaluation = self.evaluate_image(image_path, detections, metrics=metrics)

        record = {
            "image": image_path,
            "state": self.get_state(),
            "detections": detections,
            "evaluation": evaluation,
        }
        self.captured_images.append(record)
        if evaluation and evaluation.score > self.best_score:
            self.best_score = evaluation.score
            self.best_view = record
        return record

    def move_and_apply(self, delta_position, delta_orientation) -> None:
        self.move(delta_position, delta_orientation)
        self.apply_pose_to_env()

    def move_to_and_apply(self, position, orientation) -> None:
        self.move_to(position, orientation)
        self.apply_pose_to_env()
