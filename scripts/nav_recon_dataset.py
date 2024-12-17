import json
import os
from tqdm import tqdm
import imageio
import concurrent.futures
import argparse
import numpy as np
from opencv_optical_flow import compute_optical_flow_score_from_images
import h5py

def load_json(json_path):
    print("Loading json: {}".format(json_path))
    with open(json_path) as f:
        data = json.load(f)
    return data

def dump_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("Dumped to: {}".format(json_path))


data_root = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/nav_recon/vis'
clip_length = 49

meta = {
    'data_root': data_root, 
    'clip_length': clip_length, 
    'num_interval_frames': 10, 
    'fps_clip': 10, 
    'num_clips': 1, 
    'num_incomplete': 0
}

clips = []

for direction_folder in os.listdir(data_root):
    direction_folder_path = os.path.join(data_root, direction_folder)
    if not os.path.isdir(direction_folder_path):
        continue

    for vid_folder in os.listdir(direction_folder_path):
        vid_folder_path = os.path.join(direction_folder_path, vid_folder)
        if not os.path.isdir(vid_folder_path):
            continue

        # vid_path = os.path.join(vid_folder_path, 'video.mp4')
        # if not os.path.exists(vid_path):
        #     continue
        frames = os.listdir(vid_folder_path)

        if len(frames) < clip_length:
            continue
        
        frames.sort(key=lambda x: int(x.split('.')[0]))
        frames = frames[:clip_length]
        print(frames)

        clip = {
            'folder_name': os.path.join(direction_folder, vid_folder),
            'first_frame': frames[0],
            'end_frame': frames[-1],
            'flow_direction': 'Moving_Forward'
        }
        clips.append(clip)


data = {'meta': meta, 'clips': clips}
dump_json(data, '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/nav_recon/sub_samples.json')

# meta = {'data_root': '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/tmp', 'clip_length': 49, 'num_interval_frames': 10, 'fps_clip': 10, 'num_clips': 1, 'num_incomplete': 0}

# clips = [
#     {'folder_name': 'frames', 'first_frame': '000000000.jpg', 'end_frame': '000000048.jpg', 'flow_direction': 'Moving_Forward'}
# ]

# data = {'meta': meta, 'clips': clips}

# dump_json(data, '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/tmp/one_vid_ood.json')