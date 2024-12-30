import io
import os
import sys
from functools import partial
import math
import torchvision.transforms as TT
from sgm.webds import MetaDistributedWebDataset
import random
from fractions import Fraction
from typing import Union, Optional, Dict, Any, Tuple
from torchvision.io.video import av
import numpy as np
import torch
from torchvision.io import _video_opt
from torchvision.io.video import _check_av_available, _read_from_stream, _align_audio_frames
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
import decord
from decord import VideoReader
from torch.utils.data import Dataset
from tqdm import tqdm
import imageio
import json
from PIL import Image


def load_json(json_path):
    print("Loading json: {}".format(json_path))
    with open(json_path) as f:
        data = json.load(f)
    return data

def dump_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("Dumped to: {}".format(json_path))

def img_path_list_to_video(img_path_list, out_path='test.mp4', fps=10):
    writer = imageio.get_writer(out_path, fps=fps)
    for img_path in img_path_list:
        img = imageio.imread(img_path)
        writer.append_data(img)
    writer.close()


def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    # arr: T C H W
    # image_size: H W
    
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    return arr


def pad_last_frame(tensor, num_frames):
    # T, H, W, C
    if tensor.shape[0] < num_frames:
        last_frame = tensor[-int(num_frames - tensor.shape[1]) :]
        padded_tensor = torch.cat([tensor, last_frame], dim=0)
        return padded_tensor
    else:
        return tensor[:num_frames]



cmd_to_action = {
    0: "Turning_Left",
    1: "Moving_Forward",
    2: "Turning_Right",
}

# TODO: Sample weights according to action
# TODO: Improve data loading: load clip as a dict, rather than each attributes separately
class nuPlanDataset(Dataset):

    def __init__(self, 
                data_dir, 
                video_size, 
                fps, 
                max_num_frames, 
                skip_frms_num=0,   # TODO: Set to 7
                prefix_prompt="",
                n_repeat_of_actions=None,
                n_fut_traj_points=8,
                p_mask_out_heading=0,
                p_drop_action_caption=0,
                traj_key='traj_fut',
                n_subset=None,  # 30
                ind_subset=None,  # 0,...,29
                **kwargs):
        """
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """
        super(nuPlanDataset, self).__init__()

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
        self.traj_key = traj_key

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
            
            # sample_path_tuple = (data_root, clip['folder_name'], clip['first_frame'], clip['end_frame'])
            if 'img_seq' in clip:
                sample_seq = clip['img_seq']
            else:
                sample_seq = clip['img_seq_his'] + clip['img_seq_fut']
            raw_caption = clip.get(caption_key, "")  # include static and highly static
            if isinstance(raw_caption, int):
                raw_caption = cmd_to_action[raw_caption]

            fut_traj = clip[self.traj_key][:self.n_fut_traj_points]  # TODO: replace traj_fut key.

            token_key = 'lidar_pc_token' if 'lidar_pc_token' in clip else 'token'
            lidar_pc_token = clip[token_key]
            
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

        img_path_list = img_path_list[skip_frms_num:]   # skip some frames

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
            "lidar_pc_token": lidar_pc_token,
        }
        return item

    def __len__(self):
        return len(self.captions_list)

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(data_dir=path, **kwargs)
