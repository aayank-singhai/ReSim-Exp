import os
import math
import random
from torchvision.io.video import av
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
from data_utils import *

# * TODO: Improve loading samples, per video clip, not per attribute
class nuScenesDataset(Dataset):

    def __init__(self, 
                data_dir,
                video_size, 
                fps, 
                max_num_frames, 
                skip_frms_num=0,   # TODO: Set to 7 for nuplan, 0 for waymo
                prefix_prompt="",
                n_repeat_of_actions=None,
                n_fut_traj_points=8,
                p_mask_out_heading=0,
                p_drop_action_caption=0,
                reshape_mode='center',
                n_subset=None,  # 30
                ind_subset=None,  # 0,...,29
                **kwargs):
        """
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """
        super(nuScenesDataset, self).__init__()

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
        self.length = len(self.captions_list)
        self.p_mask_out_heading = p_mask_out_heading
        self.p_drop_action_caption = p_drop_action_caption
        self.reshape_mode = reshape_mode

        self.load_data_json(data_dir, n_subset=n_subset, ind_subset=ind_subset)

    # * Repeat data here
    def load_data_json(self, data_json, n_subset=None, ind_subset=None):
        infos = load_json(data_json)
        data_root = infos['meta']['data_root']
        self.data_root = data_root

        clip_infos = infos['clips']
        if n_subset is not None and ind_subset is not None:
            print("Using subset: {}/{}".format(ind_subset, n_subset))
            length_per_subset = math.ceil(len(clip_infos) / n_subset)
            start_ind = ind_subset * length_per_subset
            end_ind = (ind_subset + 1) * length_per_subset
            clip_infos = clip_infos[start_ind:end_ind]

        caption_key = "cmd"

        for clip in tqdm(clip_infos):
            
            sample_seq = clip['img_seq']
            raw_caption = clip.get(caption_key, "")  # include static and highly static
            fut_traj = clip['traj_fut'][:self.n_fut_traj_points]

            token_key = 'lidar_pc_token' if 'lidar_pc_token' in clip else 'token'
            lidar_pc_token = clip[token_key]
            if not isinstance(lidar_pc_token, str):
                lidar_pc_token = str(lidar_pc_token)
            
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

        tensor_frms = []
        for img_path in img_path_list:
            if not os.path.exists(img_path):
                print("Image not found: {}".format(img_path))
                # raise FileNotFoundError, "Image not found: {}".format(img_path)
            image = Image.open(img_path)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = np.array(image)  # H, W, C
            image = torch.from_numpy(image)
            tensor_frms.append(image)
            # image = torch.from_numpy(image).permute(2, 0, 1) # [C, H, W]
        tensor_frms = torch.stack(tensor_frms, dim=0)  # T, H, W, C
        
        tensor_frms = pad_last_frame(
            tensor_frms, max_num_frames
        )  # the len of indices may be less than num_frames, due to round error\
        tensor_frms = tensor_frms.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
        tensor_frms = resize_for_rectangle_crop(tensor_frms, video_size, reshape_mode="center")
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
