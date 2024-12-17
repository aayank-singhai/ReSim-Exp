import os
import json
import argparse
import copy
import time

import numpy as np
import yaml
from tqdm import tqdm
import torch

from utils.meta_v2 import CustomizedPairedDataSourceV2, VID_METRICS, IMG_METRICS
from utils.easydict import EasyDict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default="configs.yaml", help='the path to yaml file')
    parser.add_argument('--output', type=str, default="/cpfs01/shared/opendrivelab/opendrivelab_hdd/GenAD_Proj/eval/results/", help='the path to output folder')
    args = parser.parse_args()
    configs = EasyDict(yaml.load(open(args.yaml, "r"), Loader=yaml.FullLoader))
    device = torch.device(f'cuda:{configs.gpu}' if torch.cuda.is_available() else 'cpu')

    print("\n\n==========================================================")
    print("[TIMESTAMP] {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    print()

    if configs.metric.lower() in VID_METRICS:
        mode = 'video'
    elif configs.metric.lower() in IMG_METRICS:
        mode = 'image'
    else:
        raise ValueError('Unknown metric: {}'.format(configs.metric))

    print('prepare metrics...')
    if (configs.metric.lower() == "fvd") or (configs.metric.lower() == "kvd"):
        # from utils.fvd import FVD_KVD
        from utils.fvd_v2 import FVD_KVD
        metric = FVD_KVD(device=device)
    elif (configs.metric.lower() == "fid"):
        from utils.fid import FID
        metric = FID(device=device)
    elif (configs.metric.lower() == "clipsim"):
        from utils.clip_sim import CLIPSIM
        metric = CLIPSIM(device=device)
    elif (configs.metric.lower() == "is"):
        from utils.iscore import IScore
        metric = IScore(device=device)
    elif (configs.metric.lower() == "cmd_fvd"):
        from utils.cmd_fvd import cmd_FVD
        metric = cmd_FVD(device=device)

    print('\ncollect dataset...')
    # paired_dataset = CustomizedPairedDataSource(configs.paired_dataset, mode)
    paired_dataset = CustomizedPairedDataSourceV2(configs.paired_dataset, mode)

    # real GT
    paired_dataset.gt()
    print("\nuploading ground truth...")
    metric.update_gt(paired_dataset, **configs)

    # generated data
    paired_dataset.gen()
    print("\nevaluating...")
    metric(paired_dataset, **configs)  # * averaged over all data

    print("\n\n==========================================================")
    print("[TIMESTAMP] {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))