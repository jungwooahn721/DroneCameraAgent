# callback functions to interacti with blender scene environment
# refer to the files run_blender_drone.py (uses bpy) and dronecore/drone.py, etc.

def blender_move_callback():
    # just use drone.move() to move the drone in blender scene
    raise NotImplementedError()
    
def blender_get_pose_callback():
    # read drone pose from blender info
    raise NotImplementedError()
    
def blender_render_callback():
    # read drone pose. camera pose == drone pose since camera pose follows drone strictly
    # then render the camera view
    raise NotImplementedError()
    
