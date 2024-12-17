import json
import yaml
from tqdm import tqdm
import argparse
import os
import cv2
from tqdm import tqdm


def load_json(json_path):
    print("Loading json: {}".format(json_path))
    with open(json_path) as f:
        data = json.load(f)
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

        # if frame_ind not in total_indices:
        #     frame_ind += 1
        #     continue

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
    # import pdb; pdb.set_trace()

    # save_root = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/navsim_eval'
    save_root = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/navsim_eval/debug2'

    folder_name = os.path.basename(output_folder)
    save_name = os.path.join(save_root, folder_name)
    save_name = save_name + ".json"
    
    data = {}
    clips = []

    n_subfolders = len(os.listdir(output_folder))
    # os.walk: get all files with .mp4 extension
    N_GEN = 1000
    for i, (root, dirs, files) in enumerate(os.walk(output_folder)):
        DEBUG = True
        if DEBUG:
            if i > N_GEN: break
        print(f"Processing {i}/{n_subfolders}...")
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

    data['meta'] = dict(data_root="/cpfs01/shared/opendrivelab/opendrivelab_hdd/nuplan/dataset/nuplan-v1.1/sensor_blobs")
    data['clips'] = clips
    dump_json(data, save_name)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--folder_path", type=str, required=True)
    # args = parser.parse_args()
    # json_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/navsim_eval/debug/infer_nuplan5_lora_not-contained_all_tokens_resume-from-256_not-apply-traj_planning-11-01-14-30.json'
    # json_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/navsim_eval/infer_nuplan5_lora_not-contained_all_tokens_resume-from-256_not-apply-traj_planning-11-01-14-30.json'
    
    # json_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/navsim/dict_token2info_test_all.json'
    # json_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/navsim/token2info_test_all_list.json'
    # json_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/waymo/waymo_val_traj_cmd.json'

    # json_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/navsim_eval/debug2/infer_nuplan5_lora_not-contained_all_tokens_resume-from-256_not-apply-traj_planning-11-01-14-30_out_traj_eval_video_idm_planner_trans.json'

    # data = load_json(json_path)
    # import pdb; pdb.set_trace()

    # folder_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/outputs/infer_nuplan5_lora_not-contained_all_tokens_resume-from-256_not-apply-traj_planning-11-01-14-30'
    # folder_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/outputs/reward_infer_nuplan5_lora_resume-from-256_wm_pred-traj-12-04-11-27'
    folder_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/outputs/reward_infer_nuplan5_lora_resume-from-256_wm_gt-traj-12-04-11-23'

    make_navsim_json_from_folder(folder_path)