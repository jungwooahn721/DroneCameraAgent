# callback functions to interact with the real world environment
# refer to the files run_real_drone.py and dronecore/drone.py, etc.

def real_move_callback():
    # move the real drone using drone API
    # then update the drone pose accordingly
    raise NotImplementedError()
    
def real_get_pose_callback():
    # just get the drone pose from the real drone system
    raise NotImplementedError()
     
def real_render_callback():
    # no need to get pose since real drone provides real camera view
    # just capture the camera view from the real drone
    raise NotImplementedError()