import json
import os
from tqdm import tqdm
import imageio
import concurrent.futures
import argparse
import numpy as np

cmd_str_to_ind = {
    "Turning_Left"  : 0,
    "Moving_Forward": 1,
    "Turning_Right" : 2,
}

def load_json(json_path):
    print("Loading json: {}".format(json_path))
    with open(json_path) as f:
        data = json.load(f)
    return data

def dump_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("Dumped to: {}".format(json_path))



waymo_json = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/gaoshenyuan/mess/Waymo_svd_val_cmd_sub.json'
waymo_data = load_json(waymo_json)
import pdb; pdb.set_trace()



waymo_json = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/waymo/waymo_val_traj_cmd.json'

waymo_data = load_json(waymo_json)
waymo_clips = waymo_data['clips']
cmd_cnts = [0, 0, 0]

for clip in waymo_clips:
    cmd = clip['cmd']
    cmd_cnts[cmd_str_to_ind[cmd]] += 1

print("Command counts: {}".format(cmd_cnts))

