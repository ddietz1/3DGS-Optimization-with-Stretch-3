from scipy.spatial.transform import Rotation as R

# The robot's actual measured home orientation in map (from your tf2_echo)
home_rot = R.from_quat([0.002, 0.709, 0.012, 0.706])

# Desired relative tilt from home -- e.g., 30 degrees "pitch down" in the
# camera's OWN frame (rotate about the camera's local x/left axis, not world x)
relative_tilt = R.from_euler('x', -30, degrees=True)

# Compose: apply the relative tilt in the camera's local frame
target_rot = home_rot * relative_tilt
q = target_rot.as_quat()
print(q)