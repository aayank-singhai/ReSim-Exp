import os
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
from data_utils import *


# * TODO: Improve loading samples, per video clip, not per attribute
class SharedDataset(Dataset):

    def __init__(self, 
                data_dir, 
                video_size, 
                fps, 
                max_num_frames, 
                skip_frms_num=0,   # TODO: Set to 7 for nuplan, 0 for others
                prefix_prompt="",
                n_repeat_of_actions=None,
                n_fut_traj_points=8,
                p_mask_out_heading=0,
                p_drop_action_caption=0,
                traj_key='traj_fut',
                reshape_mode='center',
                n_subset=None,  # 30
                ind_subset=None,  # 0,...,29

                # * For nuplan only
                token_json=None,
                scene_tensor_json_folder=None,
                **kwargs):
        """
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """
        super(SharedDataset, self).__init__()

        self.video_list = []
        self.captions_list = []
        self.num_frames_list = []
        self.fps_list = []
        self.fut_traj_list = []
        self.lidar_pc_token_list = []
        
        self.video_size = video_size
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frms_num = int(skip_frms_num)
        self.prefix_prompt = prefix_prompt

        self.n_repeat_of_actions = n_repeat_of_actions

        self.n_fut_traj_points = n_fut_traj_points
        # self.length = len(self.captions_list)
        self.p_mask_out_heading = p_mask_out_heading
        self.p_drop_action_caption = p_drop_action_caption
        self.traj_key = traj_key
        self.reshape_mode = reshape_mode

        # * For nuplan only
        self.token_json = token_json
        self.scene_tensor_json_folder = scene_tensor_json_folder

        self.load_data_json(data_dir, n_subset=n_subset, ind_subset=ind_subset)

        self.length = len(self.captions_list)  # * Should be after loading data

    # * Repeat data here
    def load_data_json(self, data_json, n_subset=None, ind_subset=None):
        infos = load_json(data_json)
        data_root = infos['meta']['data_root']
        self.data_root = data_root

        clip_infos = infos['clips']
        if self.token_json is not None:
            token_json = load_json(self.token_json)
            token_keep = token_json.keys()
            clip_infos = [clip for clip in clip_infos if clip['lidar_pc_token'] in token_keep]

        if n_subset is not None and ind_subset is not None:
            print("Using subset: {}/{}".format(ind_subset, n_subset))
            length_per_subset = math.ceil(len(clip_infos) / n_subset)
            start_ind = ind_subset * length_per_subset
            end_ind = (ind_subset + 1) * length_per_subset
            clip_infos = clip_infos[start_ind:end_ind]

        print("Using {} clips".format(len(clip_infos)))

        caption_key = "cmd"

        for clip in tqdm(clip_infos):
            
            # sample_seq = clip['img_seq']
            if 'img_seq' in clip:
                sample_seq = clip['img_seq']
            else:
                sample_seq = clip['img_seq_his'] + clip['img_seq_fut']
            
            raw_caption = clip.get(caption_key, "")  # include static and highly static
            if isinstance(raw_caption, int):
                raw_caption = cmd_to_action[raw_caption]
            fut_traj = clip[self.traj_key][:self.n_fut_traj_points]

            token_key = 'lidar_pc_token' if 'lidar_pc_token' in clip else 'token'
            lidar_pc_token = clip[token_key]
            if not isinstance(lidar_pc_token, str):
                lidar_pc_token = str(lidar_pc_token)

            if self.scene_tensor_json_folder is not None:
                scene_tensor_json = os.path.join(self.scene_tensor_json_folder, lidar_pc_token + '.json')
                scene_tensor_json = load_json(scene_tensor_json, verbose=False)
                scene_tensor = scene_tensor_json[lidar_pc_token]['tensor']
                scene_tensor_valid = scene_tensor_json[lidar_pc_token]['validity']
                scene_tensor = torch.from_numpy(np.array(scene_tensor))  # [129, 21, 8]
                scene_tensor_valid = torch.from_numpy(np.array(scene_tensor_valid))  # [129, 21]
                scene_tensor_valid = scene_tensor_valid.unsqueeze(2)  # [129, 21, 1]
                scene_tensor = torch.cat([scene_tensor, scene_tensor_valid], dim=2)  # [129, 21, 9]
                scene_tensor = scene_tensor.permute(1, 0, 2)  # [21, 129, 9]   # * First dim is time, second number of agents
                scene_tensor = scene_tensor.reshape(scene_tensor.shape[0], -1)  # flatten [21, 1161]
                # !! Use fut_traj as scene_tensor
                fut_traj = scene_tensor[4: 12]
                assert fut_traj.shape[0] == self.n_fut_traj_points

            
            sample_caption = raw_caption

            sample_caption = sample_caption.replace("_", " ").lower()  # * MINOR FIX: Converting action to lowercase
            sample_caption = sample_caption[0].upper() + sample_caption[1:]
            if not sample_caption.endswith("."):
                sample_caption += "."

            if self.n_repeat_of_actions is not None:
                n_repeat = self.n_repeat_of_actions[raw_caption]
            else:
                n_repeat = 1

            # * Repeat to achieve data weighting
            sample_seq = [sample_seq] * n_repeat
            sample_caption = [sample_caption] * n_repeat
            sample_traj = [fut_traj] * n_repeat
            sample_token = [lidar_pc_token] * n_repeat

            self.video_list.extend(sample_seq)
            self.captions_list.extend(sample_caption)
            self.fut_traj_list.extend(sample_traj)
            self.lidar_pc_token_list.extend(sample_token)

    def read_img_list(self, img_path_list):
        video_size, fps, max_num_frames, skip_frms_num = \
            self.video_size, self.fps, self.max_num_frames, self.skip_frms_num
        
        img_path_list = img_path_list[skip_frms_num:]   # skip some frames
        
        tensor_frms = load_image_list_to_tensors(img_path_list)
        tensor_frms = torch.stack(tensor_frms, dim=0)  # T, H, W, C
        
        tensor_frms = pad_last_frame(
            tensor_frms, max_num_frames
        )  # the len of indices may be less than num_frames, due to round error\
        tensor_frms = tensor_frms.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
        tensor_frms = resize_for_rectangle_crop(tensor_frms, video_size, reshape_mode=self.reshape_mode)
        tensor_frms = (tensor_frms - 127.5) / 127.5
        return max_num_frames, tensor_frms

    def __getitem__(self, index):
        while True:
            video_paths = self.video_list[index]
            img_list = [
                os.path.join(self.data_root, filename) for filename in video_paths
            ]
            try:
                num_frames, video_clip = self.read_img_list(img_list)
                break
            except Exception as e:
                print("Broken data, skipping: {}".format(video_paths[-1]))
                index = random.randint(0, self.length - 1)
                continue

        # Add prefix
        caption = self.captions_list[index]
        prefix_prompt = self.prefix_prompt
        if prefix_prompt != "":
            prefix_prompt = prefix_prompt.strip()
            prefix_prompt = prefix_prompt[0].upper() + prefix_prompt[1:]
            if not prefix_prompt.endswith("."):
                prefix_prompt += "."

            if self.p_drop_action_caption > 0 and random.random() < self.p_drop_action_caption:
                caption = prefix_prompt
            else:
                caption = prefix_prompt + " " + caption

        # Traj
        fut_traj = self.fut_traj_list[index]
        fut_traj = torch.tensor(fut_traj, dtype=torch.float32)  # [8, 3]
        if self.p_mask_out_heading > 0 and random.random() < self.p_mask_out_heading:
            fut_traj[:, -1] = 0  # mask out the heading
        # Lidar pc token
        lidar_pc_token = self.lidar_pc_token_list[index]

        item = {
            "with_traj": True,
            "mp4": video_clip,
            "txt": caption,
            "num_frames": num_frames,
            "fps": self.fps,  # ? What's the use of fps?
            "fut_traj": fut_traj,
            "lidar_pc_token": lidar_pc_token
        }
        return item

    def __len__(self):
        return len(self.captions_list)

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(data_dir=path, **kwargs)
