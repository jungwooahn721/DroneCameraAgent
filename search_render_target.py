import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import bpy
from mathutils import Vector, Euler


class DroneCamera:
    def __init__(self, scene, camera, limits=None):
        self.scene = scene
        self.camera = camera
        
    def get_transform(self):
        return self.camera.location.copy(), self.camera.rotation_euler.copy()
    
    def set_transform(self, location, rotation):
        self.camera.location = location
        self.camera.rotation_euler = rotation
        self.scene.update_tag()
        bpy.context.view_layer.update()

    def move(self, delta_pos, delta_rot):
        self.camera.location += Vector(delta_pos)
        self.camera.rotation_euler.x += delta_rot[0]
        self.camera.rotation_euler.y += delta_rot[1]
        self.camera.rotation_euler.z += delta_rot[2]
        self.scene.update_tag()
        bpy.context.view_layer.update()

def parse_args():
    parser = argparse.ArgumentParser(description="Drone camera search for target.")
    parser.add_argument("--input_scene", default="assets/Koky_LuxuryHouse_0.blend")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--target_prompt", required=True, help="Text prompt for the target")
    parser.add_argument("--camera_pos", nargs=3, type=float, help="Initial camera position")
    parser.add_argument("--camera_dir", nargs=3, type=float, help="Initial camera direction (Euler angles in degrees)")
    parser.add_argument("--seed", type=int, default=721)
    parser.add_argument("--focal_length", type=float, default=50.0)
    parser.add_argument("--sensor_width", type=float, default=36.0)
    parser.add_argument("--sensor_height", type=float, default=24.0)
    parser.add_argument("--resolution", nargs=2, type=int, default=[512, 512])
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--step_size", type=float, default=0.5, help="Movement step size")
    
    # Handle blender args
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = sys.argv[1:]
        
    return parser.parse_args(argv)

def main():
    args = parse_args()