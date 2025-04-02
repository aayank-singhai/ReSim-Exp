import json

import numpy as np

meta_path = "/cpfs01/shared/opendrivelab/opendrivelab_hdd/GenAD_Proj/ad_datasets/Waymo/kitti_format/validation/meta"

raw_file = "Waymo_svd_val.json"
with open(raw_file, "r") as raw_json:
    raw_samples = json.load(raw_json)
cmd_samples = list()

for sample in raw_samples:
    gt_trajectory = np.zeros((5, 2), np.float64)
    current_index_txt = sample["first_frame"].split(".")[0]
    current_pose_file = f"{meta_path}/pose/{current_index_txt}.txt"
    current_global_pose = np.zeros((4, 4), np.float64)
    for i, line in enumerate(open(current_pose_file)):
        current_global_pose[i, :] = np.array([float(x) for x in line.split()])
    current_ego_pose = np.linalg.inv(current_global_pose)

    origin = None
    for k in range(1, 5):
        future_index = int(current_index_txt) + k * 5 - 1
        future_index_txt = f"{future_index:07d}"
        future_pose_file = f"{meta_path}/pose/{future_index_txt}.txt"
        future_global_pose = np.zeros((4, 4), np.float64)
        for i, line in enumerate(open(future_pose_file)):
            future_global_pose[i, :] = np.array([float(x) for x in line.split()])
        future_ego_pose = current_ego_pose.dot(future_global_pose)
        origin = np.array(future_ego_pose[:3, 3])
        gt_trajectory[k, :] = [-origin[1], origin[0]]

    sample.update({"traj": gt_trajectory.flatten().tolist()})

    # x-axis is positive forwards, y-axis is positive to the left, different from nuScenes
    if origin[1] >= 2:  # turn left
        sample.update({"cmd": 1})
    elif origin[1] <= -2:  # turn right
        sample.update({"cmd": 0})
    elif origin[0] <= 2:  # stop
        sample.update({"cmd": 2})
    else:  # go straight
        sample.update({"cmd": 3})

    cmd_samples.append(sample)

cmd_file = "Waymo_svd_val_cmd.json"
with open(cmd_file, "w") as cmd_json:
    json.dump(cmd_samples, cmd_json, indent=4)
