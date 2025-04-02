import json
import yaml
from tqdm import tqdm
import os
import cv2
import pickle

def load_pickle(pickle_path):
    print("Loading pickle: {}".format(pickle_path))
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data

def list_to_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("Dumped to: {}".format(json_path))


data = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/navsim/navsafe/navsafe.pkl'
data = load_pickle(data)
list_to_json(data, '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/navsim/navsafe/navsafe_tokens.json')
import pdb; pdb.set_trace()