import json
import yaml
from tqdm import tqdm

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





navsim = load_json("/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/navsim/token2info_train_list.json")
act_dist = count_actions(navsim)
navsim['meta']['action_distribution'] = act_dist
dump_json(navsim, "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/navsim/token2info_train_list.json")
# navsim train: {'Moving_Forward': 54414, 'Turning_Right': 9128, 'Turning_Left': 21567}
import pdb; pdb.set_trace()



log_split = parse_yaml("/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/navsim/default_train_val_test_log_split.yaml")
train_logs = log_split['train_logs']
val_logs = log_split['val_logs']

# navsim = load_json("/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/navsim/token2info_all.json")
navsim = load_json("/cpfs01/shared/opendrivelab/opendrivelab_hdd/tianhaochen/token2info_all.json")
navsim_train = convert_dict_to_list(navsim, train_logs)

navsim = load_json("/cpfs01/shared/opendrivelab/opendrivelab_hdd/tianhaochen/token2info_all.json")
navsim_val = convert_dict_to_list(navsim, val_logs)

dump_json(navsim_train, "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/navsim/token2info_train_list.json")
dump_json(navsim_val, "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/navsim/token2info_val_list.json")

navsim_train = load_json("/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/navsim/token2info_train_list.json")
# import pdb; pdb.set_trace()

# dump_json(nuplan_list, "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/nuplan/token2info_all_list.json")

# nuplan_new = load_json("/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/nuplan/token2info_all_list.json")

# import pdb; pdb.set_trace()