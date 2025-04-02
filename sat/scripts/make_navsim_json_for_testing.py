import json
import yaml
import argparse
import os
import cv2
from tqdm import tqdm
import math

def load_json(json_path):
    print("Loading json: {}".format(json_path))
    with open(json_path) as f:
        data = json.load(f)
    print("Successfully loaded json: {}".format(json_path))
    return data


def dump_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("Dumped to: {}".format(json_path))

def parse_yaml(yaml_path):
    print("Parsing yaml: {}".format(yaml_path))
    with open(yaml_path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

def video_to_images(video_path):
    token_name = video_path.split("_")[-2]
    output_folder = os.path.dirname(video_path)
    output_folder = os.path.join(output_folder, token_name)
    os.makedirs(output_folder, exist_ok=True)
    image_paths = []

    # Convert video to images
     # Open the video file
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ind = 0

    # * 2 hz
    past_indices = [3, 8]  # ind 8 is the current frame
    future_indices = [i for i in range(13, n_frames, 5)]
    total_indices_2hz = past_indices + future_indices

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame as image
        image_path = os.path.join(output_folder, f"frame_{frame_ind:04d}.png")
        cv2.imwrite(image_path, frame)
        image_paths.append(image_path)
        frame_ind += 1

    cap.release()
    image_paths_2hz = [image_paths[i] for i in total_indices_2hz]  # * Generated video, with compression

    return token_name, image_paths_2hz

def make_navsim_json_from_folder(output_folder):
    token2info = load_json('/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/navsim/dict_token2info_test_all.json')

    save_root = output_folder

    folder_name = os.path.basename(output_folder)
    save_name = os.path.join(save_root, folder_name)
    save_name = save_name + ".json"
    
    data = {}
    clips = []

    # n_samples_rough = 13000 

    # os.walk: get all files with .mp4 extension

    total_walks = len(list(os.walk(output_folder)))
    ind_sample = 0

    for i, (root, dirs, files) in tqdm(enumerate(os.walk(output_folder)), total=total_walks):
        DEBUG = False  # !!! DEBUG
        if DEBUG:
            N_GEN = 1000
            if i > N_GEN: break

        for file in files:
            if file.endswith(".mp4") and "Sample" in file:
                clip = dict()
                video_path = os.path.join(root, file)
                token_name, image_paths = video_to_images(video_path)

                clip['token'] = token_name

                # * Load GT image, need data root.
                clip['img_seq_his'] = token2info[token_name]['img_seq_his']
                clip['img_seq_fut'] = token2info[token_name]['img_seq_fut']
                clip['cmd']         = token2info[token_name]['cmd']
                     
                # * Load generated images, no need for data root
                clip['img_seq_gen_2hz'] = image_paths  # * Generated images, used for planning
                clip['gt_traj_fut'] = token2info[token_name]['traj_fut']
                clips.append(clip)

                ind_sample += 1
                
                # print(f"Processing {ind_sample}/{n_samples_rough}...")

    data['meta'] = dict(data_root="/cpfs01/shared/opendrivelab/opendrivelab_hdd/nuplan/dataset/nuplan-v1.1/sensor_blobs")
    data['clips'] = clips
    dump_json(data, save_name)


def merge_group_jsons(group_folder):
    # group_folder:
    # e.g.: GROUP_navsim_full_main2_plan_30k_steps

    json_paths = []
    # for root, dirs, files in os.walk(group_folder):
    #     for file in files:
    #         if file.startswith("GROUP") and file.endswith(".json"):
    #             json_paths.append(os.path.join(root, file))

    for split_folder in os.listdir(group_folder):
        split_folder_basename = os.path.basename(split_folder)
        split_folder_json = os.path.join(group_folder, split_folder, f"{split_folder_basename}.json")
        json_paths.append(split_folder_json)
    
    data = {}
    data['meta'] = load_json(json_paths[0])['meta']
    clips = []
    for json_path in json_paths:
        data_i = load_json(json_path)
        clips += data_i['clips']
    data['clips'] = clips
    print("Merged clips: ", len(clips))
    group_folder_name = os.path.basename(group_folder)
    save_path = os.path.join(group_folder, f"{group_folder_name}.json")
    dump_json(data, save_path)


def split_json(json_path):
    data = load_json(json_path)
    clips = data['clips']
    n_clips = len(clips)
    n_splits = 10
    # n_clips_per_split = n_clips // n_splits
    n_clips_per_split = math.ceil(n_clips / n_splits)
    sum_clips = 0
    for i in range(n_splits):
        split_clips = clips[i*n_clips_per_split:(i+1)*n_clips_per_split]
        sum_clips += len(split_clips)
        split_data = dict(meta=data['meta'], clips=split_clips)
        split_save_path = json_path.replace(".json", f"_split_{i}.json")
        dump_json(split_data, split_save_path)
    print("Sum clips: ", sum_clips)
    assert sum_clips == n_clips


# # merge_group_jsons("/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/outputs_hdd/GROUP_navsim_full_main2_plan_30k_steps")
# split_json("/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/outputs_hdd/GROUP_navsim_full_main2_plan_30k_steps/GROUP_navsim_full_main2_plan_30k_steps.json")
# import pdb; pdb.set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True)
    args = parser.parse_args()

    print(f"Processing folder: {args.folder_path}")
    
    make_navsim_json_from_folder(args.folder_path)