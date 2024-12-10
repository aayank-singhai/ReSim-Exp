from PIL import Image
import os
import copy

from tqdm import tqdm
import numpy as np
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModelWithProjection
import torch

from utils.easydict import EasyDict
from utils.meta import gpu_state, Metric, open_and_resize

NUM_FRAMES = 6

def load_clip_model(device):
    print("Loading CLIP...")
    clip_model = CLIPVisionModelWithProjection.from_pretrained("/cpfs01/shared/opendrivelab/GenAD_Datasets/huggingface_models/openai/clip-vit-large-patch14")
    clip_model.to(device).eval()
    processor = CLIPProcessor.from_pretrained("/cpfs01/shared/opendrivelab/GenAD_Datasets/huggingface_models/openai/clip-vit-large-patch14")
    clip = EasyDict({
        "model": clip_model,
        "processor": processor
    })
    return clip


def clipsim(paths, clip, max_sample=-1, batch_size=8, device="cuda:0", shuffle=True):
    root = paths["root"]
    length = len(paths["source"])
    interval = paths["freq"] // 2
    max_sample = max_sample if max_sample > 0 else length
    if max_sample > length:
        print("Warning: max_sample ({}) is larger than the number of clips ({}). Changing to ({})... ".format(max_sample, length, length))
        max_sample = length
    indices = np.arange(length)
    if shuffle:
        np.random.shuffle(indices)

    gen_startid = paths["gen_startid"] if "gen_startid" in paths.keys() else 0

    clipsim_temporal = []
    clipsim_consistency = []
    with torch.no_grad():                    # IMPORTANT! Otherwise, model will save the grad, leading to OOM
        for idx in tqdm(indices[:max_sample]):
            datum = paths["source"][idx]
            embs = None
            for start in range(0, len(datum), interval * batch_size):
                end = min(start + interval * batch_size, len(datum))
                videos = []
                for i in range(start, end, interval):
                    frame = datum[i]
                    videos.append(np.asarray(open_and_resize(os.path.join(root, frame))))
                
                inputs = clip.processor(images=videos, return_tensors="pt", padding=True).to(device)
                # emb = clip.model.get_image_features(**inputs).cpu()
                emb = clip.model(**inputs).image_embeds.cpu()
                embs = emb if embs is None else torch.cat([embs, emb], dim=0)
            
            sims = torch.cosine_similarity(embs[gen_startid:-1], embs[(gen_startid+1):], dim=-1)
            clipsim_temporal.append(torch.mean(sims).item())
            sims = torch.cosine_similarity(embs[0], embs[gen_startid:(gen_startid+NUM_FRAMES)], dim=-1)
            clipsim_consistency.append(torch.mean(sims).item())

    return np.mean(clipsim_temporal).item(), np.mean(clipsim_consistency).item()


def clipsim_eval(paths, max_sample=-1, batch_size=8, device="cuda:0", shuffle=True):
    clip = load_clip_model(device)
    return clipsim(paths, clip, max_sample=max_sample, batch_size=batch_size, device=device, shuffle=shuffle)


class CLIPSIM(Metric):
    def __init__(self, device):
        super().__init__("CLIPSIM")
        self.clip = load_clip_model(device)
        self.device = device

    def forward(self, gen_paths=None, max_sample=-1, batch_size=8, shuffle=True, **kwargs):
        clipsim_t, clipsim_c = clipsim(gen_paths, self.clip, max_sample=max_sample, batch_size=batch_size, 
                                       device=self.device, shuffle=shuffle)
        print('CLIPSIM temporal: {}'.format(clipsim_t))
        print('CLIPSIM consistency: {}'.format(clipsim_c))
        return clipsim_t, clipsim_c

    def update_gt(self, gt_paths=None, **kwargs):
        print("CLIPSIM is not a cross-reference metric. No need to update ground truth. Omitting...")