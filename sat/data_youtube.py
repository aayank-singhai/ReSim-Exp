import os
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
from data_utils import *
from data_share import SharedDataset

# * TODO: Improve loading samples, per video clip, not per attribute
class YouTubeDataset(SharedDataset):

    def __init__(self, 
                merge_static=False,
                exclude_highly_static=False,
                **kwargs):
        """
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """
        
        self.merge_static = merge_static
        self.exclude_highly_static = exclude_highly_static
        super(YouTubeDataset, self).__init__(**kwargs)



    # * Repeat data here
    def load_data_json(self, data_json, n_subset=None, ind_subset=None):
        infos = load_json(data_json)
        data_root = infos['meta']['data_root']
        clip_infos = infos['clips']
        if n_subset is not None and ind_subset is not None:
            print("Using subset: {}/{}".format(ind_subset, n_subset))
            length_per_subset = math.ceil(len(clip_infos) / n_subset)
            start_ind = ind_subset * length_per_subset
            end_ind = (ind_subset + 1) * length_per_subset
            clip_infos = clip_infos[start_ind:end_ind]

        for clip in tqdm(clip_infos):
            
            sample_path_tuple = (data_root, clip['folder_name'], clip['first_frame'], clip['end_frame'])
            raw_caption = clip.get("flow_direction", "")  # include static and highly static
            
            if self.exclude_highly_static and "Highly_Static" in raw_caption:
                continue

            sample_caption = raw_caption
            if self.merge_static and "Static" in sample_caption:
                sample_caption = "Moving_Forward"  # merging static and highly static to forward

            sample_caption = sample_caption.replace("_", " ").lower()  # * MINOR FIX: Converting action to lowercase
            sample_caption = sample_caption[0].upper() + sample_caption[1:]
            if not sample_caption.endswith("."):
                sample_caption += "."

            if self.n_repeat_of_actions is not None:
                n_repeat = self.n_repeat_of_actions[raw_caption]
            else:
                n_repeat = 1

            # * Repeat to achieve data weighting
            sample_path_tuple = [sample_path_tuple] * n_repeat
            sample_caption = [sample_caption] * n_repeat

            self.video_list.extend(sample_path_tuple)
            self.captions_list.extend(sample_caption)


    def __getitem__(self, index):
        while True:
            video_path = self.video_list[index]
            
            data_root, folder_name, first_frame, end_frame = video_path
            img_list = get_frame_list(first_frame, end_frame)
            img_list = [
                os.path.join(data_root, folder_name, n) for n in img_list
            ]
            try:
                num_frames, video_clip = self.read_img_list(img_list)
                break
            except Exception as e:
                print("Broken data, skipping: {}".format(video_path))
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


        item = {
            "with_traj": False,   # TODO: Use with_traj to maskout pseudo-traj of OpenDV in training
            "mp4": video_clip,
            "txt": caption,
            "num_frames": num_frames,
            "fps": self.fps,  # ? What's the use of fps?
            "fut_traj": torch.zeros((8, 3)),  # * Placeholder, not used, no traj actually,
            "lidar_pc_token": str(index),  # * Placeholder to align with nuplan dataset
        }
        return item
