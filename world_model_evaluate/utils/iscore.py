import os
from PIL import Image
import copy

import numpy as np
from tqdm import tqdm
import torch
from torchmetrics.image.inception import InceptionScore as IS

from utils.meta import Metric, open_and_resize

UCF101_PATH = "/cpfs01/shared/opendrivelab/opendrivelab_hdd/GenAD_Proj/public_datasets/UCF-101"


def is_update_from_path(paths, iscore, max_sample=-1, batch_size=8, device="gpu:0", shuffle=True):
    root = paths["root"]
    length = len(paths["source"])
    if max_sample < 0:
        max_sample = length
    elif max_sample > length:
        print("Warning: max_sample ({}) is larger than the number of clips ({}). Changing to ({})... ".format(max_sample, length, length))
        max_sample = length
    indices = np.arange(length)
    if shuffle:
        np.random.shuffle(indices)
    
    for start in tqdm(range(0, max_sample, batch_size)):
        end = min(start + batch_size, max_sample)
        img_list = []
        for idx in indices[start: end]:
            img = np.asarray(open_and_resize(os.path.join(root, paths["source"][idx])))      # (h, w, c)
            img = torch.from_numpy(copy.deepcopy(img)).permute(2, 0, 1).unsqueeze(0).to(device)        # (1, c, h, w)
            if (len(img_list) == 0) or (img.shape == img_list[0].shape):
                img_list.append(img)
            else:
                iscore.update(img)

        img_list = torch.cat(img_list, dim=0)                                           # (b, c, h, w)
        iscore.update(img_list)
        img_list = []


def is_update_from_UCF101(iscore, max_sample=-1, batch_size=8, device="gpu:0", shuffle=True):
    import cv2
    root = UCF101_PATH
    print("collecting videos...")
    paths = []
    for folder in os.listdir(root):
        for video in os.listdir(os.path.join(root, folder)):
            paths.append(os.path.join(root, folder, video))

    paths.sort()
    length = len(paths)
    if max_sample < 0:
        max_sample = length
    elif max_sample > length:
        print("Warning: max_sample ({}) is larger than the number of clips ({}). Changing to ({})... ".format(max_sample, length, length))
        max_sample = length
    
    indices = np.arange(length)
    if shuffle:
        np.random.shuffle(indices)
    for idx in tqdm(indices[:max_sample]):
        video = paths[idx]
        cap = cv2.VideoCapture(video)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(device)
                frames.append(frame)
            else:
                break
        cap.release()
        frames = torch.cat(frames, dim=0)
        iscore.update(frames)


def is_eval(paths, feature='logits_unbiased', max_sample=-1, batch_size=8, device="cuda:0", shuffle=True):
    iscore = IS(feature=feature).to(device)
    is_update_from_path(paths, iscore, max_sample=max_sample, batch_size=batch_size, device=device, shuffle=shuffle)
    return iscore.compute()


class IScore(Metric):
    def __init__(self, feature='logits_unbiased', device="cuda:0"):
        super().__init__("IScore")
        self.device = device
        self.iscore = IS(feature=feature).to(device)

    def forward(self, gen_paths=None, max_sample=-1, batch_size=8, device="cuda:0", shuffle=True, **kwargs):
        is_update_from_path(gen_paths, self.iscore, max_sample=max_sample, batch_size=batch_size, device=device, shuffle=shuffle)
        iscore_mean, iscore_std = self.iscore.compute()
        print('IS: {}'.format(iscore_mean))
        return iscore_mean, iscore_std

    def update_gt(self, gt_paths=None, **kwargs):
        print("IScore is not a cross-reference metric. No need to update ground truth. Omitting...")


if __name__ == "__main__":
    iscore = IS(feature='logits_unbiased').to("cuda:0")
    is_update_from_UCF101(iscore, max_sample=20480, batch_size=8, device="cuda:0", shuffle=True)
    print(iscore.compute())