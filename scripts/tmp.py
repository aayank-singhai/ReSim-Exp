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

video_to_images('/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/tmp/OOD.mp4', frame_rate=10)