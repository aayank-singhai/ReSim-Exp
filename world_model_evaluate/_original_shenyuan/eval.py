import os
import json
import argparse
import copy

import numpy as np
import yaml
from tqdm import tqdm
import torch

from utils.meta import DataSource, VID_METRICS, IMG_METRICS
from utils.easydict import EasyDict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default="configs.yaml", help='the path to yaml file')
    parser.add_argument('--output', type=str, default="/cpfs01/shared/opendrivelab/opendrivelab_hdd/GenAD_Proj/eval/results/", help='the path to output folder')
    args = parser.parse_args()
    configs = EasyDict(yaml.load(open(args.yaml, "r"), Loader=yaml.FullLoader))
    device = torch.device(f'cuda:{configs.gpu}' if torch.cuda.is_available() else 'cpu')

    if configs.metric.lower() in VID_METRICS:
        mode = 'video'
    elif configs.metric.lower() in IMG_METRICS:
        mode = 'image'
    else:
        raise ValueError('Unknown metric: {}'.format(configs.metric))

    print('prepare metrics...')
    if (configs.metric.lower() == "fvd") or (configs.metric.lower() == "kvd"):
        from utils.fvd import FVD_KVD
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

    print('collect dataset...')
    jsons = []
    if "path" not in configs.fake:
        for file in os.listdir(configs.fake.root):
            if not file.endswith(".json"):
                continue
            jsons.append(file)
    else:
        jsons.append(configs.fake.path)
    print("{} Datasets to be eval.".format(len(jsons)))

    real = DataSource(configs.real, mode=mode)
    print("GT: {}".format(real["root"]))
    for idx in range(len(jsons)):
        print("prepare data {}".format(idx))
        fake = copy.deepcopy(configs.fake)
        fake["path"] = os.path.join(fake.root, jsons[idx])
        print(fake["path"])
        fake = DataSource(fake, mode=mode)

        print("evaluating...")
        metric.update_gt(real, **configs)
        metric(fake, **configs)