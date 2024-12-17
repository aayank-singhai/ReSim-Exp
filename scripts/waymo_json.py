import json
import os
from tqdm import tqdm
import imageio
import concurrent.futures
import argparse
import numpy as np
from opencv_optical_flow import compute_optical_flow_score_from_images

def load_json(json_path):
    print("Loading json: {}".format(json_path))
    with open(json_path) as f:
        data = json.load(f)
    return data

def dump_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("Dumped to: {}".format(json_path))

def exist_file(file_path):
    return os.path.exists(file_path)

def get_frame_list(start_frame, end_frame):
    # eg:
    # start_frame: "000000000.jpg"
    # end_frame: "000000039.jpg"
    # return: ["000000000.jpg", "000000001.jpg", ..., "000000039.jpg"]
    ext = start_frame.split('.')[-1]
    len_num = len(start_frame.split('.')[0])
    start_frame = int(start_frame.split('.')[0])
    end_frame = int(end_frame.split('.')[0])
    frame_list = []
    for i in range(start_frame, end_frame + 1):
        frame_list.append(str(i).zfill(len_num) + f'.{ext}')
    return frame_list

  # writer = imageio.get_writer('test.mp4', fps=fps)
    # for img in img_list:
    #     writer.append_data(img)
    # writer.close()

def create_youtube_json(clip_length=49, is_train=True, is_debug=False, interval=10):
    # interval = 10  # * 1s
    # new_val: use 1080p val set (newly downloaded)

    DATA_ROOT = '/cpfs01/shared/opendrivelab/GenAD_Datasets/YouTube/'

    if is_train:
        json_path = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/gaoshenyuan/YouTube_svd.json'  # * Train
    else:
        # json_path = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/gaoshenyuan/YouTube_svd_val.json'  # * Val
        json_path = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/yangjiazhi/DVGen/data_json/YouTube_svd_val_1080p.json'  # * New Val 1080p
    infos = load_json(json_path)

    if is_debug:
        infos = infos[:1000]

    out_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/youtube_json'
    
    if is_debug:
        out_path = os.path.join(out_path, 'debug.json')
    else:
        out_path = os.path.join(out_path, os.path.basename(json_path))
        # if new_val:
        #     out_path = out_path.replace("val.json", "val_1080p.json")
    
    clip_infos = dict()
    clip_infos['meta'] = {
        'data_root': DATA_ROOT,
        'clip_length': clip_length,
        'num_interval_frames': interval,
        'fps_clip': 10,
    }
    clip_infos['clips'] = []
    n_incomplete = 0

    for frame_ind, frame in tqdm(enumerate(infos[::interval])):
        folder_name = frame['folder_name']
        # if new_val:
        #     folder_name = folder_name.replace("val_images", "1080p_val_images")

        frame_name = frame['first_frame']
        start_str, ext_str = frame_name.split('.')
        format_length = len(start_str)
        start_ind = int(start_str)

        # * Filtered out incomplete clips
        successful_clip = True

        start_frame_ind = frame_ind * interval
        end_frame_ind = start_frame_ind + clip_length - 1
        if end_frame_ind >= len(infos):
            successful_clip = False
            break
        else:
            end_frame = infos[end_frame_ind]
            end_folder_name = end_frame['folder_name']
            # if new_val:
                # end_folder_name = end_folder_name.replace("val_images", "1080p_val_images")

            end_frame_name = end_frame['first_frame']
            end_str = end_frame_name.split('.')[0]
            end_ind = int(end_str)
            if end_folder_name != folder_name or end_ind != start_ind + clip_length - 1:
                n_incomplete += 1
                successful_clip = False

        if successful_clip:
            end_name = str(start_ind + clip_length - 1).zfill(format_length) + '.' + ext_str
            clip_info = {
                'folder_name': folder_name,
                'first_frame': frame_name,
                'end_frame': end_name,
            }
            clip_infos['clips'].append(clip_info)

    print("len(clip_infos): {}".format(len(clip_infos['clips'])))

    clip_infos['meta']['num_clips'] = len(clip_infos['clips'])
    clip_infos['meta']['num_incomplete'] = n_incomplete

    dump_json(clip_infos, out_path)

def create_waymo_json(clip_length=49, is_debug=False, interval=1):
    # new_val: use 1080p val set (newly downloaded)

    # DATA_ROOT = '/cpfs01/shared/opendrivelab/GenAD_Datasets/YouTube/'
    DATA_ROOT = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/GenAD_Proj/ad_datasets/Waymo/kitti_format/validation/images_0'

    json_path = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/GenAD_Proj/ad_datasets/Waymo/kitti_format/validation/waymo_val_all.json'
    # if is_train:
    #     json_path = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/gaoshenyuan/YouTube_svd.json'  # * Train
    # else:
    #     # json_path = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/gaoshenyuan/YouTube_svd_val.json'  # * Val
    #     json_path = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/yangjiazhi/DVGen/data_json/YouTube_svd_val_1080p.json'  # * New Val 1080p
    infos = load_json(json_path)

    # import pdb; pdb.set_trace()
    infos = infos['videos'][0]['full_list']

    if is_debug:
        infos = infos[:1000]

    out_path = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/waymo'

    # out_path = os.path.join(out_path, 'debug.json')
    
    if is_debug:
        out_path = os.path.join(out_path, 'debug.json')
    else:
        out_path = os.path.join(out_path, os.path.basename(json_path))
    
    clip_infos = dict()
    clip_infos['meta'] = {
        'data_root': DATA_ROOT,
        'clip_length': clip_length,
        'fps_clip': 10,  # * Waymo is also 10hz: https://waymo.com/open/about/
    }
    clip_infos['clips'] = []
    n_incomplete = 0

    for frame_ind, frame_name in tqdm(enumerate(infos)):
        # folder_name = frame['folder_name']
        # frame_name = frame['first_frame']

        start_str, ext_str = frame_name.split('.')
        format_length = len(start_str)
        start_ind = int(start_str)

        # * Filtered out incomplete clips
        successful_clip = True

        start_frame_ind = frame_ind * interval    # TODO: Why should we multiply interval 10??
        end_frame_ind = start_frame_ind + clip_length - 1
        if end_frame_ind >= len(infos):
            successful_clip = False
            break
        else:
            end_frame_name = infos[end_frame_ind]
            end_str = end_frame_name.split('.')[0]
            end_ind = int(end_str)
            if end_ind != start_ind + clip_length - 1:
                n_incomplete += 1
                successful_clip = False

        if successful_clip:
            end_name = str(start_ind + clip_length - 1).zfill(format_length) + '.' + ext_str
            clip_info = {
                'first_frame': frame_name,
                'end_frame': end_name,
            }
            clip_infos['clips'].append(clip_info)

    print("len(clip_infos): {}".format(len(clip_infos['clips'])))

    clip_infos['meta']['num_clips'] = len(clip_infos['clips'])
    clip_infos['meta']['num_incomplete'] = n_incomplete

    dump_json(clip_infos, out_path)

def create_waymo_traj_and_cmd(json_path, is_debug=False, n_past=9, fut_horizon=8, n_traj_channels=3, interval=1):
    meta_path = "/cpfs01/shared/opendrivelab/opendrivelab_hdd/GenAD_Proj/ad_datasets/Waymo/kitti_format/validation/meta"

    with open(json_path, "r") as raw_json:
        raw = json.load(raw_json)
        raw_clips = raw["clips"]
    
    raw_clips = raw_clips[::interval]  # * interval between each clips
    if is_debug:
        raw_clips = raw_clips[:100]
    cmd_clips = list()
    raw['meta']['interval'] = interval  # * 0.1 s

    for clip_ind, clip in tqdm(enumerate(raw_clips)):
        gt_trajectory = np.zeros((fut_horizon, n_traj_channels), np.float64)
        
        # current_index_txt = clip["first_frame"].split(".")[0]

        first_frame_name, ext_str = clip["first_frame"].split(".")[0], clip["first_frame"].split(".")[1]
        len_name = len(first_frame_name)
        first_frame_ind = int(first_frame_name)
        cur_frame_ind = first_frame_ind + n_past - 1
        current_index_txt = str(cur_frame_ind).zfill(len_name)

        current_pose_file = f"{meta_path}/pose/{current_index_txt}.txt"
        current_global_pose = np.zeros((4, 4), np.float64)
        for i, line in enumerate(open(current_pose_file)):
            current_global_pose[i, :] = np.array([float(x) for x in line.split()])
        current_ego_pose = np.linalg.inv(current_global_pose)

        origin = None
        for k in range(0, fut_horizon):
            # future_index = int(current_index_txt) + k * 5 - 1   # * 10 hz -> 2 hz
            future_index = int(current_index_txt) + (k + 1) * 5  # * Bug here? - fixed

            # future_index_txt = f"{future_index:07d}"
            future_index_txt = str(future_index).zfill(len_name)

            future_pose_file = f"{meta_path}/pose/{future_index_txt}.txt"
            future_global_pose = np.zeros((4, 4), np.float64)
            for i, line in enumerate(open(future_pose_file)):
                future_global_pose[i, :] = np.array([float(x) for x in line.split()])
            future_ego_pose = current_ego_pose.dot(future_global_pose)
            origin = np.array(future_ego_pose[:3, 3])
            # * Vista
            # gt_trajectory[k, :2] = [-origin[1], origin[0]]

            # * GenADv3
            gt_trajectory[k, :2] = [origin[0], origin[1]]

        # clip.update({"traj": gt_trajectory.flatten().tolist()})
        clip.update({"traj_fut": gt_trajectory.tolist()})
        clip['img_seq'] = get_frame_list(clip['first_frame'], clip['end_frame'])
        clip['token'] = clip_ind

        # * Vista Cmds
        # x-axis is positive forwards, y-axis is positive to the left, different from nuScenes
        # if origin[1] >= 2:  # turn left
        #     clip.update({"cmd": 1})
        # elif origin[1] <= -2:  # turn right
        #     clip.update({"cmd": 0})
        # elif origin[0] <= 2:  # stop
        #     clip.update({"cmd": 2})
        # else:  # go straight
        #     clip.update({"cmd": 3})

        # * Waymo Cmds
        if origin[1] >= 2:  # turn left
            clip.update({"cmd": "Turning_Left"})
        elif origin[1] <= -2:  # turn right
            clip.update({"cmd": "Turning_Right"})
        # elif origin[0] <= 2:  # stop
        #     clip.update({"flow_direction": 2})
        # else:  # go straight
        #     clip.update({"flow_direction": 3})
        else:
            clip.update({"cmd": "Moving_Forward"})

        # cmd_samples.append(sample)
        cmd_clips.append(clip)
    
    raw['meta']['num_clips'] = len(cmd_clips)
    raw['clips'] = cmd_clips  # * Update the clips

    print("Num clips: {}".format(len(cmd_clips)))

    cmd_file = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/waymo/waymo_val_traj_cmd.json"
    with open(cmd_file, "w") as cmd_json:
        json.dump(raw, cmd_json, indent=4)

def traverse_waymo(json_path):
    data = load_json(json_path)
    clips = data['clips']
    for i, clip in enumerate(clips):
        clip['token'] = i

    data['clips'] = clips
    out_path = json_path.replace(".json", "_token.json")
    dump_json(data, out_path)

if __name__ == '__main__':
    # nuplan_json = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/navsim/token2info_test_all_list.json'

    # youtube_raw_json = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/yangjiazhi/GenADv3/data_json/YouTube_svd_val_1080p.json'
    # youtube_json = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/youtube_json/YouTube_svd_val_1080p_clip-len-49_interval-10_77k_flow.json'  # * Val verified.
    # youtube_json = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/youtube_json/YouTube_svd_clip-len-49_interval-10_5M_flow_round2.json'

    # nuplan = load_json(nuplan_json)

    # youtube_raw = load_json(youtube_raw_json)
    # youtube = load_json(youtube_json)
    # import pdb; pdb.set_trace()


    # waymo_json = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/waymo/waymo_val_traj_cmd.json'
    # waymo = load_json(waymo_json)
    # import pdb; pdb.set_trace()

    # traverse_waymo('/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/waymo/waymo_val_traj_cmd.json')
    # import pdb; pdb.set_trace()

    INTERVAL = 5  # * interval 2hz for first frames
    create_waymo_traj_and_cmd('/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/waymo/waymo_val_all.json', is_debug=False, interval=5)
    import pdb; pdb.set_trace()