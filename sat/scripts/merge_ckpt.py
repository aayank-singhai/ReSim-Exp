import json
import yaml
from tqdm import tqdm
import os
import cv2

cmd_to_action = {
    0: "Turning_Left",
    1: "Moving_Forward",
    2: "Turning_Right",
}

def load_json(json_path):
    print("Loading json: {}".format(json_path))
    with open(json_path) as f:
        data = json.load(f)
    return data

def parse_yaml(yaml_path):
    print("Parsing yaml: {}".format(yaml_path))
    with open(yaml_path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def convert_dict_to_list(data, logs=None):
    print("Converting dict to list")
    
    out = dict()
    out['meta'] = {
        'data_root': '/cpfs01/shared/opendrivelab/opendrivelab_hdd/nuplan/dataset/nuplan-v1.1/sensor_blobs',
        'len_img_seq_his': 16,
        'len_img_seq_fut': 40,
        'len_traj_his': 4,
        'len_traj_fut': 10,  
        'fps_clip': 10,
        # 'num_clips': 103288,
        'notes': 'Last frame of img_seq_his is the current frame, the first one in img_seq_fut is the next frame, only use the first 8 traj in traj_fut.'
    }
    
    clips = []
    num_clips = 0
    for token, sample in tqdm(data.items()):
        sample['lidar_pc_token'] = token
        sample['cmd'] = cmd_to_action[sample['cmd']]

        log_name = sample['log_name']
        if log_name not in logs:
            continue
        clips.append(sample)
        num_clips += 1
    out['clips'] = clips
    out['meta']['num_clips'] = num_clips
    return out

def count_actions(data):
    clips = data['clips']
    actions = {}
    for clip in clips:
        action = clip['cmd']
        if action not in actions:
            actions[action] = 0
        actions[action] += 1
    print(actions)
    return actions


def dump_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("Dumped to: {}".format(json_path))

import cv2
import os

def video_to_images(vid_path, frame_rate):
    # Create an output folder based on the video file name
    output_folder = os.path.dirname(vid_path)
    output_folder = os.path.join(output_folder, "frames")
    os.makedirs(output_folder, exist_ok=True)
    image_paths = []

    # Open the video file
    cap = cv2.VideoCapture(vid_path)
    original_frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame based on the specified frame rate
        if frame_count % int(original_frame_rate // frame_rate) == 0:
            image_path = os.path.join(output_folder, f"{saved_frame_count:09d}.jpg")
            cv2.imwrite(image_path, frame)
            image_paths.append(image_path)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_frame_count} frames from {vid_path}")

    return image_paths

import torch


# * Merge ckpt1 into ckpt2 with a weighted ratio
def merge_two_ckpts(ckpt1_path, ckpt2_path, ckpt1_ratio, output_path):
    """
    Merges two PyTorch checkpoints with a weighted ratio for shared keys.
    Keeps additional keys from ckpt2.

    Args:
        ckpt1_path (str): Path to the first checkpoint file.
        ckpt2_path (str): Path to the second checkpoint file.
        ckpt1_ratio (float): Weight for ckpt1 (0 <= ckpt1_ratio <= 1).
        output_path (str): Path to save the merged checkpoint.

    Returns:
        None
    """
    # Load checkpoints
    print("Start loading checkpoints...")
    ckpt1 = torch.load(ckpt1_path, map_location="cpu")
    ckpt2 = torch.load(ckpt2_path, map_location="cpu")

    print(f"Checkpoint loaded!")

    merged_ckpt = ckpt2.copy()  # Start with ckpt2, keeping all its keys
    ckpt1_module = ckpt1["module"]  # Check if ckpt1 is a DataParallel model
    ckpt2_module = ckpt2["module"]  # Check if ckpt2 is a DataParallel model

    # import pdb; pdb.set_trace()

    # # Merge only shared keys
    # for key in tqdm(ckpt1.keys()):
    #     if key in ckpt2 and isinstance(ckpt1[key], torch.Tensor) and isinstance(ckpt2[key], torch.Tensor):
    #         merged_ckpt[key] = ckpt1[key] * ckpt1_ratio + ckpt2[key] * (1 - ckpt1_ratio)
    #     else:
    #         print("Key {} is not shared, directly save it".format(key))
    #         merged_ckpt[key] = ckpt1[key]  # Preserve non-shared keys from ckpt1

    # Merge only shared keys in module
    for key in tqdm(ckpt1_module.keys()):
        if key in ckpt2_module and isinstance(ckpt1_module[key], torch.Tensor) and isinstance(ckpt2_module[key], torch.Tensor):
            try:
                ckpt2_module[key] = ckpt1_module[key] * ckpt1_ratio + ckpt2_module[key] * (1 - ckpt1_ratio)
            except:
                print("Key {} is have different shape, skip itq".format(key))
            print("Merged key: {}".format(key))
        else:
            print("Key {} is not shared, directly save it".format(key))
            ckpt2_module[key] = ckpt1_module[key]  # Preserve non-shared keys from ckpt1

    merged_ckpt['module'] = ckpt2_module

    # Save merged checkpoint
    torch.save(merged_ckpt, output_path)
    print(f"Merged checkpoint saved to {output_path}")

# Example usage
# merge_two_ckpts("checkpoint1.pth", "checkpoint2.pth", 0.7, "merged_checkpoint.pth")

ckpt1 = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/tmp_ckpts/merge_ckpts/main5_joint_stage2_high_small-lr_full-12-14-07-46/30000/mp_rank_00_model_states.pt'
ckpt2 = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/tmp_ckpts/merge_ckpts/gpu32_main6_joint_stage2_high_w_carla_human_drive_token-02-20-11-37/20000/mp_rank_00_model_states.pt'
merge_two_ckpts(ckpt1, ckpt2, 0.9, '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/tmp_ckpts/merge_ckpts/merged.pt')