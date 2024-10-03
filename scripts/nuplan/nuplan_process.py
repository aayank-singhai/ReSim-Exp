import json

import json
import os
from tqdm import tqdm
import imageio
import numpy as np

def load_json(json_path):
    print("Loading json: {}".format(json_path))
    with open(json_path) as f:
        data = json.load(f)
    return data

def dump_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("Dumped to: {}".format(json_path))

# Split the json file into multiple json files
def split_json_path(json_path, k_split=5):
    # Read the original JSON file
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Determine the number of items per split
    total_items = len(data)
    items_per_split = total_items // k_split

    # Create a directory to store the split JSON files
    base_dir = os.path.dirname(json_path)
    split_dir = os.path.join(base_dir, 'split_json')
    os.makedirs(split_dir, exist_ok=True)

    # Split the data into multiple chunks and save as separate JSON files
    for i in range(k_split):
        start_index = i * items_per_split
        end_index = (i + 1) * items_per_split

        # Handle the last chunk to include any remaining items
        if i == k_split - 1:
            end_index = total_items

        # Generate the split data
        split_data = data[start_index:end_index]

        # Create the split file name
        old_base_name = os.path.basename(json_path).split('.')[0]
        split_filename = old_base_name + f'_split_{i}.json'
        split_filepath = os.path.join(split_dir, split_filename)

        # Write the split data to a separate JSON file
        with open(split_filepath, 'w') as split_file:
            json.dump(split_data, split_file, indent=2)

        print(f'Split {i + 1} saved to {split_filepath}')


def merge_json_dir(json_dir, frame_rate=10):
    json_files = [
        f for f in os.listdir(json_dir) if f.startswith('full') and f.endswith('.json')
    ]
    
    print("Total number of json files: ", len(json_files))

    def get_ind(filename):
        # full_020_desc_blip2.json
        return int(filename.split('_')[1])
    
    json_files.sort(key=get_ind)
    print("After sorted, the first 5 files are: ", json_files[:5])

    # Initialize an empty list to store the data from each JSON file
    merged_data = []
    total_cnt = 0

    # Read each JSON file and append its data to the merged_data list
    for json_file in tqdm(json_files):
        json_path = os.path.join(json_dir, json_file)
        data = load_json(json_path)
        total_cnt += int(data['meta']['total_counts'])

        merged_data.extend(data["annotations"])

    merged = dict()
    merged['meta'] = {
        'frame_rate': frame_rate,
        'total_counts': total_cnt
    }
    merged['annotations'] = merged_data

    merged_json_path = os.path.join('/cpfs01/user/yangjiazhi/workspace/Ask-Anything/video_chat/youtube_process', 'full_merged.json')
    dump_json(merged_data, merged_json_path)
    print(f'Merged JSON file saved to: {merged_json_path}')
    print(f'Total counts: {total_cnt}')


def merge_two_json(json_path_1, json_path_2, out_path):
    json1 = json.load(open(json_path_1))
    json2 = json.load(open(json_path_2))

    json1.extend(json2)
    
    with open(out_path, 'w') as file:
        json.dump(json1, file, indent=2)
    
    print(f'Merged JSON file saved to: {out_path}')

# /cpfs01/shared/opendrivelab/GenAD_Datasets/YouTube/V1/_output/full_desc_blip2/full_merged_30788590frames_10hz.json
# Create singapore-all json
def create_subset_json_specified_on_name(full_json_path, selected_youtuber='', selected_video_subname='', save_root='./'):
    # If selected_youtuber / selected_video_subname is not specified, then add all without selection.
    full_json = load_json(full_json_path)
    full_data = full_json['annotations']
    selected_data = []
    selected_videos = []
    for example in full_data:
        user_name = example['folder_path'].split('/')[0]
        video_name = example['folder_path'].split('/')[1]
        if selected_youtuber in user_name \
                and selected_video_subname in video_name \
                    and video_name not in set(selected_videos):  # Not selected before
            selected_videos.append(video_name)

    print(f"Selected {len(selected_videos)} Videos:\n{selected_videos}")

    selected_train_all = []
    selected_val_all = []
    for selected_video_name in tqdm(selected_videos):
        selected_data = []
        for example in full_data:
            user_name = example['folder_path'].split('/')[0]
            video_name = example['folder_path'].split('/')[1]
            if selected_youtuber in user_name \
                    and selected_video_name in video_name:
                selected_data.append(example)
        train_val_split = 0.85
        len_train = int(len(selected_data) * train_val_split)
        selected_train, selected_val = selected_data[:len_train], selected_data[len_train:]
        print(f"Adding {len(selected_train)} train and {len(selected_val)} val from {selected_video_name}.")
        selected_train_all.extend(selected_train)
        selected_val_all.extend(selected_val)

    selected_train_all_json = {
        'meta': {
            'frame_rate': 10,
            'total_counts': len(selected_train_all)
        },
        'annotations': selected_train_all
    }

    selected_val_all_json = {
        'meta': {
            'frame_rate': 10,
            'total_counts': len(selected_val_all)
        },
        'annotations': selected_val_all
    }

    save_train_path = os.path.join(save_root, f'selected_Train_{selected_youtuber}_{selected_video_subname}.json')
    save_val_path = os.path.join(save_root, f'selected_Val_{selected_youtuber}_{selected_video_subname}.json')

    dump_json(selected_train_all_json, save_train_path)
    dump_json(selected_val_all_json, save_val_path)  


def save_img_seq_to_video(out_path, img_seq, fps=2):
    # img_seq: np array
    writer = imageio.get_writer(out_path, fps=fps)
    for img in img_seq:
        writer.append_data(img)
    writer.close()

def check_one_scenario_video(scenario_index, nuplan_infos, video_root):
    img_seq = []
    for info in nuplan_infos:
        if info['scenario_index'] == scenario_index:
            front_img = os.path.join('/cpfs01/shared/opendrivelab/opendrivelab_hdd/nuplan/dataset/nuplan-v1.1/sensor_blobs/', info['front_img'])
            img_seq.append(imageio.imread(front_img))
            scenario_info = info['scenario_info']

    os.makedirs(video_root, exist_ok=True)
    video_out_path = os.path.join(video_root, f'scenario_{scenario_index}_{scenario_info}.mp4')
    save_img_seq_to_video(video_out_path, img_seq, fps=10)


def get_img_seq_start_from_index(self, start_index, start_sample):
    start_scenario = start_sample['scenario_index']
    start_frame = start_sample['frame_index']
    prev_frame = start_frame

    temporal_length = self.num_frames
    seq_indexes = [start_index + i for i in range(temporal_length)]
    
    # End of the dataset
    if any([ind >= len(self.samples) for ind in seq_indexes]):
        return None
    
    sample_seq = []
    for i, sample_ind in enumerate(seq_indexes):
        cur_sample = self.samples[sample_ind]

        cur_scenario = cur_sample['scenario_index']
        cur_frame = cur_sample['frame_index']

        # End of the video   
        if cur_scenario != start_scenario:
            return None

        if i > 0 and (not cur_frame > prev_frame):
            print(f"cur scenario: {cur_scenario}")
            print(f"Invalid frame order: cur_frame: {cur_frame}, last_frame: {prev_frame}")
            return None

        prev_frame = cur_frame
        sample_seq.append(cur_sample)

    img_paths = [
        os.path.join(self.data_root, sample['front_img']) \
                for sample in sample_seq
    ]
    img_seq = [
        self.preprocess_image(img_path)[0] for img_path in img_paths
    ]
    return img_seq

def get_clip(start_index, start_sample, all_samples, temporal_length):
    start_scenario = start_sample['scenario_index']
    start_frame = start_sample['frame_index']
    prev_frame = start_frame

    seq_indexes = [start_index + i for i in range(temporal_length)]
    
    # End of the dataset
    if any([ind >= len(all_samples) for ind in seq_indexes]):
        return None
    
    clip = {
        'scenario_index': start_scenario,
        'scenario_info': start_sample['scenario_info'],
        'img_seq': []
    }

    samples = []
    for i, sample_ind in enumerate(seq_indexes):
        cur_sample = all_samples[sample_ind]

        cur_scenario = cur_sample['scenario_index']
        cur_frame = cur_sample['frame_index']

        # End of the video   
        if cur_scenario != start_scenario:
            return None

        if i > 0 and (cur_frame != prev_frame + 1):
            print(f"cur scenario: {cur_scenario}")
            print(f"Invalid frame order: cur_frame: {cur_frame}, last_frame: {prev_frame}")
            return None

        prev_frame = cur_frame
        clip['img_seq'].append(cur_sample['front_img'])
        samples.append(cur_sample)

    cmd, _ = get_driving_command(samples[0], samples[-1])
    clip['cmd'] = cmd

    return clip

def image_to_sequence(info_json, start_interval, num_frames_per_video=8, out_path='/cpfs01/user/yangjiazhi/workspace/DiffuSim/nuplan_outputs'):
    clips = []
    start_inds = []
    samples = load_json(info_json)
    for i, sample in enumerate(samples):
        frame_index = sample['frame_index']
        if frame_index % start_interval == 0 and frame_index + num_frames_per_video - 1 <= 32:
            start_inds.append(i)
    
    for start_ind in start_inds:
        start_sample = samples[start_ind]
        clip = get_clip(start_ind, start_sample, all_samples=samples, temporal_length=num_frames_per_video)
        if clip is not None:
            clips.append(clip)

    print(f"len dataset: {len(clips)}")
    name = info_json.split('/')[-1].replace('.json', '')
    dump_json(clips, os.path.join(out_path, f'clips_of_{name}.json'))

def merge_json_dir(json_dir, out_path):
    json_files = [
        f for f in os.listdir(json_dir) if f.startswith('nuplan') and f.endswith('.json')
    ]
    
    print("Total number of json files: ", len(json_files))

    def get_ind(filename):
        # nuplan_info_0_2hz.json
        return int(filename.split('_')[2])
    
    json_files.sort(key=get_ind)
    print("After sorted, the first 5 files are: ", json_files[:5])

    merged_data = []
    total_cnt = 0

    last_scenario = 0
    for json_file in tqdm(json_files):
        json_path = os.path.join(json_dir, json_file)
        data = load_json(json_path)
        for i, d in enumerate(data):
            data[i]['scenario_index'] += (last_scenario + 1)
        last_scenario = data[-1]['scenario_index']
        total_cnt += len(data)
        merged_data.extend(data)


    # merged_json_path = os.path.join('/cpfs01/user/yangjiazhi/workspace/DiffuSim/nuplan_outputs/FULL_OUTPUT', 'full_merged.json')
    dump_json(merged_data, out_path)
    print(f'Merged JSON file saved to: {out_path}')
    print(f'Total counts: {total_cnt}')


def get_r_mat_and_t_from_matrix(matrix):
    r_mat = matrix[:3, :3]
    t = matrix[:3, 3]
    return r_mat, t

def get_driving_command(start_sample, end_sample):
    # Translate the coords of ego of end sample to that of start frame.
    # Step1: end_frame: lidar -> global
    pos_end_ego = np.array([0, 0, 0, 1])
    lidar2global_end = np.array(end_sample['lidar2global'])
    pos_end_global = lidar2global_end @ pos_end_ego.T  # in global
    
    # Step2: end_frame: global -> start_frame: ego
    lidar2global_start = np.array(start_sample['lidar2global'])
    lidar2global_r_mat, lidar2global_t = get_r_mat_and_t_from_matrix(lidar2global_start)

    global2lidar_r_mat = np.linalg.inv(lidar2global_r_mat)
    global2lidar_t = -global2lidar_r_mat @ lidar2global_t
    global2lidar = np.eye(4)
    global2lidar[:3, :3] = global2lidar_r_mat
    global2lidar[:3, 3] = global2lidar_t
    
    pos_start_ego = global2lidar @ pos_end_global  # in ego: start frame
    # scenario_index = 55  # Starting left turn   array([43.52812198, 67.66016998,  1.17415712,  1.        ])
    # scenario_index = 57  # Starting right turn    array([ 35.45794618, -91.1462343 ,  -3.44649796,   1.    ])

    offset_y = pos_start_ego[1]
    MARGIN = 2
    if offset_y >= MARGIN:
        cmd = 'left'
        print("Left turn")
    elif offset_y <= -1 * MARGIN:
        cmd = 'right'
        print("Right turn")
    else:
        cmd = 'forward'
    return cmd, pos_start_ego


if __name__ == "__main__":
    #  Merge json files
    json_dir = '/cpfs01/user/yangjiazhi/workspace/DiffuSim/nuplan_outputs/FULL_OUTPUT_2'
    merge_json_dir(json_dir, out_path='/cpfs01/user/yangjiazhi/workspace/DiffuSim/nuplan_outputs/FULL_OUTPUT_2/full_merged.json')

    # 1. Create final nuplan json
    # nuplan_json = '/cpfs01/user/yangjiazhi/workspace/DiffuSim/nuplan_outputs/FULL_OUTPUT_2/nuplan_info_0_2hz.json'
    nuplan_json = '/cpfs01/user/yangjiazhi/workspace/DiffuSim/nuplan_outputs/FULL_OUTPUT_2/full_merged.json'
    start_interval = 4
    num_frames_per_video = 8
    image_to_sequence(nuplan_json, start_interval, num_frames_per_video, out_path='/cpfs01/user/yangjiazhi/workspace/DiffuSim/nuplan_outputs/FULL_OUTPUT_2')

    
    # 2. Unitest cmd --> Checked, Correct !
    nuplan_json = '/cpfs01/user/yangjiazhi/workspace/DiffuSim/nuplan_outputs/DEBUG_OUTPUT/nuplan_info_0_2hz.json'
    # scenario_index = 55  # Starting left turn   array([43.52812198, 67.66016998,  1.17415712,  1.        ])
    # scenario_index = 57  # Starting right turn    array([ 35.45794618, -91.1462343 ,  -3.44649796,   1.    ])
    scenario_index = 0
    
    nuplan_info = load_json(nuplan_json)

    check_one_scenario_video(scenario_index, nuplan_info, video_root='/cpfs01/user/yangjiazhi/workspace/DiffuSim/nuplan_outputs/DEBUG_OUTPUT')

    scene = []
    for sample in nuplan_info:
        if sample['scenario_index'] == scenario_index:
            scene.append(sample)

    cmd, pos_start_ego = get_driving_command(scene[0], scene[8])
    print("CMD: ", cmd)
    print("pos_start_ego: ", pos_start_ego)