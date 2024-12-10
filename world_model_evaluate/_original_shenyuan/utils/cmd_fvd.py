import os
import math
from PIL import Image
import json
import copy

import numpy as np
from tqdm import tqdm
import torch

from utils.fvd_utils import get_fvd_logits, frechet_distance, polynomial_mmd
from utils.fvd import load_fvd_model, i3d_process_img_paths
from utils.pytorch_i3d import InceptionI3d
from utils.meta import I3D_PATH, MIN_I3D_TIME, Metric, open_and_resize
from utils.meta import PAIRED_DATASET_SUBSET_FLAG as SUBSET_FLAG

PRINT_WARNING = False


class cmd_FVD(Metric):
    def __init__(self, device):
        super().__init__("cmd_FVD")
        self.i3d = load_fvd_model(device)
        self.device = device
        self.embs_gt = None

    def forward(self, gen_paths=None, max_sample=-1, batch_size=8, shuffle=True, n_runs=1, self_compare=False, fix_gt=True, freq=-1, **kwargs):
        if (self.embs_gt is None) or (not fix_gt):
            self.update_gt(**kwargs)
        
        if freq < 0:
            self.freq = max(self.freq, gen_paths["freq"])
        else:
            self.freq = freq

        tmp = copy.deepcopy(gen_paths)
        source = tmp.pop("source")
        tmp["source"] = []
        cmd_gens = [ copy.deepcopy(tmp) for _ in range(self.total_cmds) ]
        for clip in source:
            cmd_gens[self.cmd_mapping[gen_paths.get_index(clip[0])]]["source"].append(clip)
        
        embs_gen = [i3d_process_img_paths(cmd_gen, self.i3d, self.freq, max_sample=max_sample, batch_size=batch_size, device=self.device, shuffle=shuffle) \
                for cmd_gen in cmd_gens]

        fvd_list = [[] for _ in range(self.total_cmds)]
        fvd_means = [0 for _ in range(self.total_cmds)]
        for cmd in range(self.total_cmds):
            print("\n==========================================================")
            print(f"[Command] {cmd}")
            for i in range(n_runs):
                # print(f'Run {i+1}/{n_runs}')
                fvd = frechet_distance(embs_gen[cmd], self.embs_gt[cmd])
                fvd_list[cmd].append(fvd.item())
                # print(f'FVD: {fvd.item():.4f}')
        
            fvd_mean = np.mean(fvd_list[cmd]).item()
            print('FVD: {}'.format(fvd_mean))
            fvd_means[cmd] = fvd_mean

        return fvd_means

    def update_gt(self, gt_paths=None, max_sample=-1, batch_size=8, shuffle=True, n_runs=1, self_compare=False, fix_gt=True, freq=-1, total_cmds=-1, **kwargs):
        assert total_cmds > 0, "please specify the total number of commands. current total_cmds: {}".format(total_cmds)
        self.total_cmds = total_cmds
        self.freq = freq if freq > 0 else gt_paths["freq"]
        self.cmd_mapping = dict()
        supp = json.load(open(gt_paths.supp, "r"))
        
        tmp = copy.deepcopy(gt_paths)
        source = tmp.pop("source")
        tmp["source"] = []
        cmd_gts = [ copy.deepcopy(tmp) for _ in range(total_cmds) ]
        for clip in source:
            clip_index = gt_paths.get_index(clip[0])
            self.cmd_mapping[clip_index] = supp[clip_index]["cmd"]
            cmd_gts[self.cmd_mapping[clip_index]]["source"].append(clip)

        self.embs_gt = [i3d_process_img_paths(cmd_gt, self.i3d, self.freq, max_sample=max_sample, batch_size=batch_size, device=self.device, shuffle=shuffle) \
                for cmd_gt in cmd_gts]
        print("GT path updated.")