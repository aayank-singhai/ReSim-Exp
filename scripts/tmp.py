import json


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

def convert_dict_to_list(data):
    print("Converting dict to list")
    
    out = dict()
    out['meta'] = {
        'data_root': '/cpfs01/shared/opendrivelab/opendrivelab_hdd/nuplan/dataset/nuplan-v1.1/sensor_blobs',
        'len_img_seq_his': 16,
        'len_img_seq_fut': 40,
        'len_traj_his': 4,
        'len_traj_fut': 10,  
        'fps_clip': 10,
        'num_clips': 103288,
        'notes': 'Last frame of img_seq_his is the current frame, the first one in img_seq_fut is the next frame, only use the first 8 traj in traj_fut.'
    }
    
    clips = []
    for token, sample in data.items():
        sample['lidar_pc_token'] = token
        sample['cmd'] = cmd_to_action[sample['cmd']]
        clips.append(sample)
    out['clips'] = clips
    return out


def dump_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("Dumped to: {}".format(json_path))


nuplan = load_json("/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/nuplan/token2info_all.json")
nuplan_list = convert_dict_to_list(nuplan)
dump_json(nuplan_list, "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/nuplan/token2info_all_list.json")

nuplan_new = load_json("/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/nuplan/token2info_all_list.json")

import pdb; pdb.set_trace()