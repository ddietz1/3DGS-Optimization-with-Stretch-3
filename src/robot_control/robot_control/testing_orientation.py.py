from scipy.spatial.transform import Rotation as Rot
import json, math

with open("stretch_user/captures/Jul-30_dataset_3/TestPoseD4353_d435_map_pose.json") as f:
    record = json.load(f)

q = record["orientation"]
R = Rot.from_quat(q)

fwd_z = R.apply([0, 0, 1])
fwd_x = R.apply([1, 0, 0])

print("yaw if forward = local +Z:", math.degrees(math.atan2(fwd_z[1], fwd_z[0])))
print("yaw if forward = local +X:", math.degrees(math.atan2(fwd_x[1], fwd_x[0])))