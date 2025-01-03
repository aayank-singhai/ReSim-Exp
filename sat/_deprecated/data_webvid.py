import os, io, csv, math, random
import numpy as np
from einops import rearrange
from decord import VideoReader

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import torchvision
import imageio
from PIL import Image
from collections import abc
from inspect import isfunction

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


def random_transform_collate_fn(batch):
    # batch is a list, len: batch_size
    # each sample in the batch is a dict
    
    # * Get common kwargs for collate_fn
    is_train = batch[0]['is_train']
    p_center_crop = batch[0]['p_center_crop']

    image = []
    caption = []
    for sample in batch:
        image.append(sample['image'])
        caption.append(sample['caption'])

    image = torch.cat(image, dim=0)
    h, w = image.shape[-2:]
    if is_train and random.random() < p_center_crop and h != w:
        image = transforms.functional.center_crop(image, output_size=min(h, w))

    image = rearrange(image, '... c h w -> ... h w c')
    collated_batch = dict(
        image=image,
        caption=caption
        # is_image.
    )

    return collated_batch


# TODO: Improve cropping scheme
# if w / h > expected_ratio:
# 	Resize(…)
# 	centercrop(…)

class WebVid10M(Dataset):
    def __init__(
            self,
            csv_path, 
            video_folder,
            sample_size=256,  # hw
            sample_stride=4,
            sample_stride_prob=None,
            sample_n_frames=16,
            is_image=False,
            is_train=True,
            split_train_val=True,
            manual_length=None,
            n_sample_per_epoch=None,
            p_original_cap=0.,  # TODO: Use original cap with p=0.25
            p_center_crop=0.,
        ):
        print(f"loading annotations from {csv_path} ...")
        with open(csv_path, 'r') as csvfile:
            # self.dataset = list(csv.DictReader(csvfile))
            dataset = list(csv.DictReader(csvfile))
            print("Successfully loaded annotations")

        self.video_folder    = video_folder
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image

        sample_stride = [sample_stride] if isinstance(sample_stride, int) else sample_stride
        if len(sample_stride) == 1:
            sample_stride_prob = [1.0]
        else:
            if sample_stride_prob is None:
                sample_stride_prob = [1.0 / len(sample_stride)] * len(sample_stride)
            else:
                assert len(sample_stride) == len(sample_stride_prob), "sample_stride and sample_stride_prob should have the same length."
        print(f"sample_stride: {sample_stride}")
        print(f"sample_stride_prob: {sample_stride_prob}")
        self.sample_stride = sample_stride
        self.sample_stride_prob = sample_stride_prob


        is_square = isinstance(sample_size, int) or (isinstance(sample_size, abc.Iterable) and sample_size[0] == sample_size[1])
        if isinstance(sample_size, int):
            sample_size = (sample_size, sample_size)

        if is_square:
             # * Square
            self.img_transforms = transforms.Compose([
                transforms.Resize(sample_size[0]),
                transforms.CenterCrop(sample_size),
                # transforms.Lambda(lambda x: rearrange(x, '... c h w -> ... h w c'))
            ])
        else:
            # * Non-square
            sample_size = tuple(sample_size)
            self.img_transforms = transforms.Compose([
                    transforms.Resize(sample_size),
                    # transforms.Lambda(lambda x: rearrange(x, '... c h w -> ... h w c'))
                ])
        self.is_square = is_square
        self.sample_size = sample_size
        self.accept_ratio = [0.8, 1.2]
        self.p_original_cap = p_original_cap
        
        filtered_dataset = dataset

        if split_train_val:
            # train-val split 0.9
            len_train = int(len(filtered_dataset) * 0.9)
            if is_train:
                filtered_dataset = filtered_dataset[:len_train]
            else:
                filtered_dataset = filtered_dataset[len_train:]
        
        # after the split, no overlap between train and val
        if manual_length is not None:
            print(f"manual length: {manual_length}")
            filtered_dataset = filtered_dataset[:manual_length]
        
        self.dataset = filtered_dataset
        self.length = len(self.dataset)
        print(f"data scale: {len(self.dataset)}")

        self.common_kwargs = dict(
            is_train=is_train,
            p_center_crop=p_center_crop
        )

        if n_sample_per_epoch is None:
            self.n_sample_per_epoch = self.length
        else:
            self.n_sample_per_epoch = min(n_sample_per_epoch, self.length)
        print(f"n_sample_per_epoch: {self.n_sample_per_epoch}")

    def get_video_seq(self, idx):
        video_dict = self.dataset[idx]
        videoid, page_dir = video_dict['videoid'], video_dict['page_dir']

        def exist_cap(cap_name):
            return cap_name in video_dict and video_dict[cap_name] != ""
        
        cap_name = 'name'
        if random.random() < self.p_original_cap:
            cap_name = 'name'
        elif exist_cap('caption_v1'):
            cap_name = 'caption_v1'
        
        caption = video_dict[cap_name]
        
        video_dir    = os.path.join(self.video_folder, page_dir, f"{videoid}.mp4")
        video_reader = VideoReader(video_dir)

        # * Get the FPS of the video
        sample_stride = random.choices(self.sample_stride, weights=self.sample_stride_prob, k=1)[0]
        video_fps = video_reader.get_avg_fps() / sample_stride
        video_length = len(video_reader)
        
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * sample_stride + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            img_idxes = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            img_idxes = [random.randint(0, video_length - 1)]

        
        pixel_values = video_reader.get_batch(img_idxes).asnumpy()  # [n_frames, H, W, C]
        pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()  # [n_frames, C, H, W]

        pixel_values = (pixel_values / 127.5 - 1.)  # [-1, 1]
        pixel_values = pixel_values.float()

        del video_reader

        if self.is_image:
            pixel_values = pixel_values[0]

        return pixel_values, caption, video_fps

    def __len__(self):
        return self.n_sample_per_epoch

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, caption, video_fps = self.get_video_seq(idx)
                if pixel_values.shape[0] < self.sample_n_frames:
                    # * abnormal length, not used in trainnig.
                    print(f"abnormal length: {pixel_values.shape[0]}")
                    idx = random.randint(0, self.length-1)
                    continue

                if not self.is_square:
                    # pixel_values.shape [8, 3, 336, 596]
                    w, h = pixel_values.shape[-1], pixel_values.shape[-2]
                    
                    # src_wh_ratio = pixel_values.shape[-1] / pixel_values.shape[-2]
                    src_wh_ratio = w / h
                    tgt_wh_ratio = self.sample_size[1] / self.sample_size[0]
                    relative_ratio = src_wh_ratio / tgt_wh_ratio
                    if relative_ratio < self.accept_ratio[0] or relative_ratio > self.accept_ratio[1]:
                        # * abnormal ratio, not used in trainnig.
                        print(f"abnormal ratio: {relative_ratio}")
                        idx = random.randint(0, self.length-1)
                        continue
                break

            except Exception as e:
                idx = random.randint(0, self.length-1)

        pixel_values = self.img_transforms(pixel_values)  # Data transforms on video sequence (not image)

        sample = dict(image=pixel_values, caption=caption, video_fps=video_fps, **self.common_kwargs)
        return sample



# if __name__ == "__main__":
#     # from animatediff.utils.util import save_videos_grid
#     dataset = WebVid10M(
#         csv_path="/cpfs01/user/yangjiazhi/workspace/MoNA_Project/VideoDatasets/results_2M_val_1/0.csv",
#         video_folder="/cpfs01/user/yangjiazhi/workspace/MoNA_Project/VideoDatasets/VideoData/webvid_sub_val/data/videos",
#         sample_size=256,
#         sample_stride=4, 
#         sample_n_frames=16,
#         # is_image=True,
#         is_image=False,
#     )
#     # dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=16,)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0)
#     print("dataloader ready")

#     for idx, batch in enumerate(dataloader):
#         import pdb; pdb.set_trace()
#         # batch['image'].shape [2, 16, 256, 256, 3]  [batch_size, n_frames, h, w, c]
#         print(batch["image"].shape, len(batch["caption"]))
#         for i in range(batch["image"].shape[0]):
#             print("saving", os.path.join(".", f"{idx}-{i}.mp4"))
#             print("caption", batch["caption"][i])
#             save_videos_grid(batch["image"][i:i+1].permute(0,2,1,3,4), os.path.join(".", f"{idx+100}-{i}.mp4"), rescale=True)
#             import pdb; pdb.set_trace()