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
    original_frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_ind = 0

    past_indices = [3, 8]  # ind 8 is the current frame
    future_indices = [i for i in range(13, n_frames, 5)]
    total_indices = past_indices + future_indices

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_ind not in total_indices:
            frame_ind += 1
            continue

        # Save frame as image
        image_path = os.path.join(output_folder, f"frame_{frame_ind:04d}.png")
        cv2.imwrite(image_path, frame)
        image_paths.append(image_path)
        frame_ind += 1

    cap.release()
    return token_name, image_paths

def make_navsim_json_from_folder(output_folder):
    save_root = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/navsim_eval'
    folder_name = os.path.basename(output_folder)
    save_name = os.path.join(save_root, folder_name)
    save_name = save_name + ".json"
    
    data = {}

    n_subfolders = len(os.listdir(output_folder))
    # os.walk: get all files with .mp4 extension
    for i, (root, dirs, files) in enumerate(os.walk(output_folder)):
        print(f"Processing {i}/{n_subfolders}...")
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(root, file)
                token_name, image_paths = video_to_images(video_path)
                data[token_name] = image_paths
                # break
        # break
    dump_json(data, save_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True)
    args = parser.parse_args()

    make_navsim_json_from_folder(args.folder_path)