# Drone Photographer

This project focuses on creating an automated drone control agent for aesthetic photography.
Our current plan in to train a model on Blender scenes and object and deploy them to real world scenarios. 

## Drone Rendering & Real-World Capture

This repo supports two execution environments:

- **Blender runtime**: render synthetic images from `.blend` scenes by running a Python script *inside Blender* (uses `bpy`).
- **Real-world runtime**: run the same core planning/annotation logic while controlling an actual drone (no `bpy`).

The shared logic lives in `dronecore/`, while environment-specific code lives in `blender/` and `real/`. Two top-level entry scripts are the only files you execute directly.

## Conceptual Outline of the Repository

- `run_blender_drone.py`  
  Top-level entrypoint executed by Blender (`blender -b -P run_blender_drone.py -- ...`)

- `run_real_drone.py`  
  Top-level entrypoint executed by Python (`python run_real_drone.py ...`)

- `dronecore/`  
  Shared core logic (no `bpy`): Drone class, pose planning/math, config, schemas, output writing

- `blender_drone/`  
  Blender-only implementation (uses `bpy`): scene loading, camera control, rendering, conversions

- `real_drone/`  
  Real-world implementation: actual drone manipulation + image capture + safety + telemetry/logging  
  - `backends/` (e.g., MAVSDK / ROS2 / simulator)
  - `capture/` (camera pipelines)
  - `safety/` (preflight + geofence)
  - `logging/` (telemetry + timestamps)

- `tests/`  
  Unit tests for core math/schemas and shared logic

