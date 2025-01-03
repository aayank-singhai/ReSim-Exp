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
from data_video_custom import SFTDataset
from data_nuplan import nuPlanDataset

def load_json(json_path):
    print("Loading json: {}".format(json_path))
    with open(json_path) as f:
        data = json.load(f)
    return data

def dump_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("Dumped to: {}".format(json_path))

def pad_last_frame(tensor, num_frames):
    # T, H, W, C
    if tensor.shape[0] < num_frames:
        last_frame = tensor[-int(num_frames - tensor.shape[1]) :]
        padded_tensor = torch.cat([tensor, last_frame], dim=0)
        return padded_tensor
    else:
        return tensor[:num_frames]

