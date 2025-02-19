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

# TODO: Improve data loading: load clip as a dict, rather than each attributes separately
class nuPlanDataset(SharedDataset):

    def __init__(self, 
                **kwargs):
        """
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """
        super(nuPlanDataset, self).__init__(**kwargs)