import json
import multiprocessing as mp
from os import listdir, path
from typing import Any, Callable, Dict

import cv2 as cv
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data import get_worker_info
from torchvision import transforms
from tqdm import tqdm


def exists(var) -> bool:
    return var is not None


def default(var, val) -> Any:
    return var if exists(var) else val


def default_worker_init_fn(worker_id: int) -> None:
    torch.manual_seed(torch.initial_seed() + worker_id)
    worker_info = get_worker_info()

    if exists(worker_info):
        dataset = worker_info.dataset
        glob_start = dataset._start
        glob_end = dataset._end

        per_worker = int((glob_end - glob_start) / worker_info.num_workers)
        worker_id = worker_info.id

        dataset._start = glob_start + worker_id * per_worker
        dataset._end = min(dataset._start + per_worker, glob_end)


class LightningDataset(LightningDataModule):
    """
    Abstract LightningDataModule that represents a dataset we can train a Lightning module on.
    """

    def __init__(
            self,
            *args,
            batch_size: int = 8,
            num_workers: int = 16,
            train_shuffle: bool = True,
            val_shuffle: bool = False,
            val_batch_size: int = None,
            worker_init_fn: Callable = None,
            collate_fn: Callable = None,
            train_sampler: Callable = None,
            test_sampler: Callable = None,
            val_sampler: Callable = None
    ) -> None:
        super(LightningDataset, self).__init__()
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        val_batch_size = default(val_batch_size, batch_size)

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle
        self.train_sampler = train_sampler
        self.test_sampler = test_sampler
        self.val_sampler = val_sampler
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn

    def train_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
        else:
            worker_init_fn = self.worker_init_fn
        return DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn
        )

    def val_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
        else:
            worker_init_fn = self.worker_init_fn
        return DataLoader(
            self.val_dataset,
            sampler=self.val_sampler,
            batch_size=self.val_batch_size,
            shuffle=self.val_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn
        )

    def test_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
        else:
            worker_init_fn = self.worker_init_fn
        return DataLoader(
            self.test_dataset,
            sampler=self.test_sampler,
            batch_size=self.val_batch_size,
            shuffle=self.val_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn
        )


class CARLADatasetTestVideo(Dataset):
    def __init__(self) -> None:
        super(CARLADatasetTestVideo, self).__init__()
        gt_path = "/cpfs01/shared/opendrivelab/opendrivelab_hdd/yangjiazhi/GenADv3/outputs_hdd/navsim_test_with_gt_traj-02-19-09-18"
        carla_path = "/cpfs01/shared/opendrivelab/opendrivelab_hdd/yangjiazhi/GenADv3/outputs_hdd/navsim_test_with_carla_traj-02-19-09-18"

        positive_samples = []
        negative_samples = []

        for sample_id in range(52):
            positive_sample_folder = path.join(gt_path, f"{sample_id}")
            negative_sample_folder = path.join(carla_path, f"{sample_id}")

            for file in listdir(positive_sample_folder):
                if file.startswith(f"Sample_folder-{sample_id}"):
                    cap = cv.VideoCapture(path.join(gt_path, positive_sample_folder, file))
                    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
                    frames = []
                    while True:
                        ret, frame = cap.read()
                        if ret:
                            # Frame was successfully read, parse it
                            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                            frame = torch.from_numpy(frame)
                            frames.append(frame)
                        else:
                            # Reach the end of video
                            break
                    cap.release()
                    assert len(frames) == total_frames
                    video = torch.stack(frames) / 255.0
                    video = rearrange(video, "t h w c -> t c h w")
                    video = F.interpolate(video, 224, mode="bicubic")

                    positive_samples.append({
                        "videos": video,
                        "rewards": torch.Tensor([1.0])
                    })

            for file in listdir(negative_sample_folder):
                if file.startswith(f"Sample_folder-{sample_id}"):
                    cap = cv.VideoCapture(path.join(carla_path, negative_sample_folder, file))
                    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
                    frames = []
                    while True:
                        ret, frame = cap.read()
                        if ret:
                            # Frame was successfully read, parse it
                            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                            frame = torch.from_numpy(frame)
                            frames.append(frame)
                        else:
                            # Reach the end of video
                            break
                    cap.release()
                    assert len(frames) == total_frames
                    video = torch.stack(frames) / 255.0
                    video = rearrange(video, "t h w c -> t c h w")
                    video = F.interpolate(video, 224, mode="bicubic")

                    negative_samples.append({
                        "videos": video,
                        "rewards": torch.Tensor([0.0])
                    })

        self.samples = positive_samples + negative_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


class CARLADataset(Dataset):
    def __init__(
            self,
            meta_file: str,
            data_aug: bool,
            start_id: int = None,
            end_id: int = None,
            extend: bool = False
    ) -> None:
        super(CARLADataset, self).__init__()
        with open(meta_file, "r") as f:
            meta_data = json.load(f)

        self.data_root = meta_data["meta"]["data_root"]
        self.clip_length = meta_data["meta"]["clip_length"]

        self.clips = []
        self.rewards = []
        self.images_cache = {}  # Cache for images

        if start_id is not None:
            items = meta_data["clips"][start_id:]
        elif end_id is not None:
            items = meta_data["clips"][:end_id]
        else:
            items = meta_data["clips"]

        # Initialize multiprocessing
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = list(tqdm(pool.imap(self.load_item, items), total=len(items)))

        # Unpack valid results
        for valid, item in results:
            if valid:
                self.images_cache.update(item["cache"])
                self.clips.append(item["img_seq"][:self.clip_length])
                self.rewards.append(item["score_penalty"])

        if extend:
            self.clips = self.clips * 10
            self.rewards = self.rewards * 10

        self.data_aug = data_aug

    def load_item(self, item):
        valid = True
        little_cache = {}
        for img in item["img_seq"][:self.clip_length]:
            img_path = path.join(self.data_root, img)
            try:
                image = Image.open(img_path)
                image = image.resize((224, 224), resample=Image.LANCZOS)
                if not image.mode == "RGB":
                    image = image.convert("RGB")
                little_cache[img_path] = image
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                valid = False
                break  # Stop processing this item if an error occurs

        return valid, {"cache": little_cache, "img_seq": item["img_seq"], "score_penalty": item["score_penalty"]}

    def __len__(self) -> int:
        return len(self.rewards)

    def __getitem__(self, idx: int) -> Dict:
        img_seq = self.clips[idx]
        images = []
        for img in img_seq:
            img_path = path.join(self.data_root, img)

            # Use cached image
            image = self.images_cache.get(img_path)
            image = transforms.ToTensor()(image)
            if self.data_aug:
                image = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)(image)
                image = transforms.GaussianBlur(kernel_size=5)(image)
            images.append(image)
        assert len(images) == self.clip_length
        video = torch.stack(images)
        return {
            "videos": video,
            "rewards": torch.Tensor([self.rewards[idx]])
        }


class LightningCARLA(LightningDataset):
    def __init__(self, meta_file: str, **kwargs) -> None:
        super(LightningCARLA, self).__init__(**kwargs)
        self.meta_file = meta_file
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = CARLADataset(meta_file=self.meta_file, data_aug=True)
            self.val_dataset = CARLADatasetTestVideo()
        elif stage == "test":
            self.test_dataset = CARLADatasetTestVideo()
        else:
            raise ValueError(f"Invalid stage: {stage}")
