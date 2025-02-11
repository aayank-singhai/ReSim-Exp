import json

import numpy as np
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from tqdm import tqdm

version = "v1.0-trainval"
data_root = "/cpfs01/shared/opendrivelab/datasets/nuscenes"
# key_frame = 5
# valid_scenes = splits.train  # train
valid_scenes = splits.val  # val

# horizon = 25
horizon = 49
past_frames = 10

current_index = 8
total_key_frame = 11
key_frame = 8   # * N future key frames
n_traj_channel = 3

def get_global_pose(rec, nusc, inverse=False):
    lidar_token = rec["data"]["LIDAR_TOP"]
    lidar_sample_data = nusc.get("sample_data", lidar_token)

    sd_ep = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    sd_cs = nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])
    if inverse:
        sensor_from_ego = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=True)
        ego_from_global = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=True)
        pose = sensor_from_ego.dot(ego_from_global)
    else:
        global_from_ego = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=False)
        ego_from_sensor = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False)
        pose = global_from_ego.dot(ego_from_sensor)
    return pose


def get_matrix(calibrated_data, inverse=False):
    output = np.eye(4)
    output[:3, :3] = Quaternion(calibrated_data["rotation"]).rotation_matrix
    output[:3, 3] = calibrated_data["translation"]
    if inverse:
        output = np.linalg.inv(output)
    return output


dataset = NuScenes(version=version, dataroot=data_root, verbose=True)
can_bus = NuScenesCanBus(dataroot=data_root)
all_samples = dataset.sample

all_samples.sort(key=lambda x: (x["scene_token"], x["timestamp"]))
valid_sample_indices = list()


for index in range(len(all_samples)):
    is_valid_data = True
    previous_sample = None
    # for t in range(key_frame):
    for t in range(total_key_frame):
        index_t = index + t

        # exceed the dataset capacity
        if index_t >= len(all_samples):
            is_valid_data = False
            break

        current_sample = all_samples[index_t]

        # check if scene is the same
        if (
                dataset.get("scene", current_sample["scene_token"])["name"] not in valid_scenes or
                (previous_sample is not None and current_sample["scene_token"] != previous_sample["scene_token"])
        ):
            is_valid_data = False
            break

        previous_sample = current_sample

    if is_valid_data:
        valid_sample_indices.append(index)

new_sample_indices = list()
new_samples = list()
for index in valid_sample_indices:
    sample_dict = dict()
    frame_files = list()
    enough_sweeps = True
    first_sample = all_samples[index]
    camera_token = first_sample["data"]["CAM_FRONT"]
    for i in range(horizon):
        if camera_token == "":
            enough_sweeps = False
            break
        camera_data = dataset.get("sample_data", camera_token)
        # if i % 6 == 0 and not camera_data["is_key_frame"]:
        #     enough_sweeps = False
        #     break
        frame_files.append(camera_data["filename"])
        previous_frame = camera_data
        camera_token = previous_frame["next"]
    if enough_sweeps:
        new_sample_indices.append(index)
        sample_dict.update({"img_seq": frame_files})
        new_samples.append(sample_dict)

clip_ind = 0
# * A little bit incorrect, but should not be a big deal
for index, sample_dict in tqdm(zip(new_sample_indices, new_samples)):
    # index: 120
    # sample_dict: {'frames': [xxx.jpg, xxx.jpg, ...]}  # one clip
    # import pdb; pdb.set_trace()

    current_index = index + (past_frames // 5)
    
    # sample = all_samples[index]
    sample = all_samples[current_index]
    
    scene_name = dataset.get("scene", sample["scene_token"])["name"]
    try:
        vehicle_monitor = can_bus.get_messages(scene_name, "vehicle_monitor")
        has_can_bus = True
    except:
        has_can_bus = False

    gt_trajectory = np.zeros((key_frame, n_traj_channel), np.float64)

    current_ego_pose = get_global_pose(sample, dataset, inverse=True)

    camera_token = sample["data"]["CAM_FRONT"]
    camera_data = dataset.get("sample_data", camera_token)
    # assert sample_dict["frames"][0] == camera_data["filename"]

    origin = None
    future_global_pose = None

    # jump to current frame before this
    for i in range(key_frame):
        # index_i = index + i
        index_i = current_index + (i + 1)

        if index_i < len(all_samples):
            future_sample = all_samples[index_i]

            future_global_pose = get_global_pose(future_sample, dataset, inverse=False)
            future_ego_pose = current_ego_pose.dot(future_global_pose)

            origin = np.array(future_ego_pose[:3, 3])
            origin[0] = -origin[0]

            # gt_trajectory[i, :2] = [origin[0], origin[1]]
            gt_trajectory[i, :2] = [origin[1], origin[0]]

            if has_can_bus:
                time_diff = None
                for each_message in vehicle_monitor:
                    if time_diff is None:
                        time_diff = abs(each_message["utime"] - future_sample["timestamp"])
                        last_message = each_message
                    else:
                        if abs(each_message["utime"] - future_sample["timestamp"]) < time_diff:
                            time_diff = abs(each_message["utime"] - future_sample["timestamp"])
                            last_message = each_message
                        else:
                            break

    sample_dict.update({"traj": gt_trajectory.tolist()})

    # cmd vista
    if origin[0] >= 2:  # turn left
        sample_dict.update({"cmd_vista": 1})
    elif origin[0] <= -2:  # turn right
        sample_dict.update({"cmd_vista": 0})
    elif origin[1] <= 2:  # stop
        sample_dict.update({"cmd_vista": 2})
    else:
        sample_dict.update({"cmd_vista": 3})

    if origin[0] >= 2:  # turn left
        sample_dict.update({"cmd": "Turning_Left"})
    elif origin[0] <= -2:  # turn right
        sample_dict.update({"cmd": "Turning_Right"})
    else:
        sample_dict.update({"cmd": "Moving_Forward"})

    sample_dict['token'] = clip_ind
    clip_ind += 1


infos = dict()
infos['meta'] = {
    'data_root': data_root,
    'clip_length': horizon,
    'fps_clip': 12,  # * nusc frame rate
    'num_clips': len(new_samples),
}
infos['clips'] = new_samples

new_anno_file = "nuScenes.json"
with open(new_anno_file, "w") as new_anno_json:
    json.dump(infos, new_anno_json, indent=4)