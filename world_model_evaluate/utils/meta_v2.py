# some global variables and default dataset settings

import os
import json
from PIL import Image

from utils.easydict import EasyDict
import cv2
from tqdm import tqdm

I3D_PATH = "/cpfs01/shared/opendrivelab/opendrivelab_hdd/qiuyihang/models/i3d/i3d_pretrained_400.pt"
MAX_BATCH = 8
TARGET_RESOLUTION = (224, 224)
MIN_I3D_TIME = 9
TARGET_WIDTH = 448
TARGET_HEIGHT = 256

VID_METRICS = ["fvd", "kvd", "clipsim", "cmd_fvd"]
IMG_METRICS = ["fid", "is"]
# PAIRED_DATASET_SUBSET_FLAG = "<SUBSET>"
PAIRED_DATASET_SUBSET_FLAG = "DOMAIN"


def video_to_images(video_path):
    output_folder = video_path.replace(".mp4", "")

    os.makedirs(output_folder, exist_ok=True)
    image_paths = []

    # Convert video to images
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_ind = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame as image
        image_path = os.path.join(output_folder, f"frame_{frame_ind:04d}.png")
        cv2.imwrite(image_path, frame)
        image_paths.append(image_path)
        frame_ind += 1

    cap.release()
    return image_paths

class CustomizedPairedDataSourceV2(EasyDict):
    # expected folder structure
    # <roots> (multiple splits)
    # | -- <root>
    # |   |-- <folder1> 
    # |   |   |-- <Sample_folder-1_token1_000000.mp4>
    # |   |   |-- <GT_folder-1_token1_000001.mp4>
    # |   |   |-- other folders
    # |   |   |-- ...
    # |   |-- <folder2> 
    # |   |   |-- <Sample_folder-2_token2_000000.mp4>
    # |   |   |-- <GT_folder-2_token2_000001.mp4>
    # |   |   |-- other folders
    # |   |   |-- ...
    
    SUBSET_FLAG = PAIRED_DATASET_SUBSET_FLAG
    DEFAULT = {
        "format": "mp4",
        # "gt_key": "real",
        # "gen_key": "virtual",
        "gt_key": "GT",
        "gen_key": "Sample",
        "rec_key": "Rec",

        "split_len": 3860,  # 166 for Waymo, 596 for nuScenes
        "supp": None
    }

    def __init__(self, initial, mode=None):
        assert isinstance(initial, dict)
        tmp = EasyDict(self.DEFAULT)
        tmp.update(initial)
        initial = tmp

        print(initial)
        
        if "mode" in initial:
            assert initial.mode in ["image", "video"]
            if mode is None:
                mode = initial.mode
        if mode is None:
            mode = "image"

        self.recollect = False
        if mode == "image":
            self.recollect = True

        assert "freq" in initial, "You must set the frequency of your dataset."
        assert "max_frameid" in initial, "You must set the number of frames per clip as `max_frameid` in your dataset."
        assert "gen_startid" in initial, "You need to specify the start id of all clips as `gen_startid` for your generated data."
        
        self.update(initial)
        self.pop("path")
        
        self.source = []
        self._sweep_for_all(has_backup = "backup_root" in initial.keys())

        print("You are loading paired dataset from [{}]".format(self.root))
        sample = self.source[0]
        if isinstance(sample, list):
            sample = sample[0]
        print("A GT sample is as follows: [{}]".format(sample.replace(self.SUBSET_FLAG, self.gt_key)))
        print("#(samples): [{}]".format(len(self.source)))

        if self.supp is not None:
            print("GT supplement is loaded from [{}]".format(self.supp))
        

    def _sweep_for_all(self, has_backup=False):
        all_clips = []

        roots = self.backup_root if has_backup else self.root  # * Could be a list

        if "GROUP" in roots.split('/')[-1]:
            roots = [
                os.path.join(roots, split) for split in os.listdir(roots)
            ]

        if not isinstance(roots, list):
            roots = [roots]

        print(f"Evaluating roots: {roots}")

        for root in roots:
            
            print(f"Evaluating Root: {root}")


            for folder in tqdm(os.listdir(root)):
                print(f"Folder: {folder}")


                # folder: 0, 1, 2, 3, ...
                collect_path = os.path.join(root, folder)

                if os.path.isfile(collect_path):
                    continue
                
                for filename in os.listdir(collect_path):
                    if filename.endswith(self.format):
                        break
                
                if not filename.endswith(self.format):
                    continue
                
                gt_filename = f"{self.gt_key}_{filename.split('_', 1)[1]}"  # GT_xxx.mp4
                gen_filename = f"{self.gen_key}_{filename.split('_', 1)[1]}"  # Sample_xxx.mp4
                
                # check if the file exists
                gt_path = os.path.join(collect_path, gt_filename)
                gen_path = os.path.join(collect_path, gen_filename)
                if not os.path.exists(gt_path) or not os.path.exists(gen_path):
                    continue

                gt_images = gt_path.replace(".mp4", "")
                gen_images = gen_path.replace(".mp4", "")
                
                if not os.path.exists(gt_images):
                    video_to_images(gt_path)
                    
                if not os.path.exists(gen_images):
                    video_to_images(gen_path)

                try:
                    assert len(os.listdir(gt_images)) == len(os.listdir(gen_images)) and len(os.listdir(gt_images)) > 0, \
                    f"The number of frames in GT: {len(os.listdir(gt_images))} and \
                        generated videos:{len(os.listdir(gen_images))} are not matched."
                except:
                    print(f"In Folder: {folder}, Re generating frames")
                    video_to_images(gt_path)
                    video_to_images(gen_path)

                assert len(os.listdir(gt_images)) == len(os.listdir(gen_images)), \
                    f"Gt_images: {gt_images}, The number of frames in GT: {len(os.listdir(gt_images))} and \
                        generated videos:{len(os.listdir(gen_images))} are not matched."
                
                clip = []
                for frame in os.listdir(gen_images):
                    clip.append(os.path.join(gen_images, frame))
                    # * Do not include root here
                    # clip.append(os.path.join(folder, gen_filename.replace(".mp4", ""), frame))
                
                # Sort 
                clip.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
                clip = [
                    f.replace(self.gen_key, self.SUBSET_FLAG) for f in clip
                ]
                
                # * Select frames to evaluate
                clip = clip[self.gen_startid: self.max_frameid]

                # import pdb; pdb.set_trace()
                all_clips.append(clip)

        self.source = all_clips

    def switch_to_subset(self, subset):
        self.paired_subset = subset

    def gt(self):
        self.paired_subset = self.gt_key
    
    def gen(self):
        self.paired_subset = self.gen_key

    def get_index(self, filename):
        # split_id = int(filename.split("/")[-4].split("split")[-1])
        # scene_id = int(filename.split("/")[-1].split("_")[1])
        # return split_id * self.split_len + scene_id
        # filename: one filename of one frame in a video clip
        index = int(filename.split('/')[-2].split('_')[-2])

        # import pdb; pdb.set_trace()  # TODO: Check this?
        return index

class DataSource(EasyDict):
    # To use `DataSource` class, you must generate a `summary.json` for your generated datasets and GT datasets.
    # Otherwise, please use `CustomizedDataSource`.
    NUSCENES_VAL = {
        "root": "/cpfs01/shared/opendrivelab/nuscenes",
        "path": "/cpfs01/shared/opendrivelab/opendrivelab_hdd/GenAD_Proj/eval/dataset_source/nuscenes/nuscenes_val_<MODE>.json",
    }
    YOUTUBE_VAL = {
        "root": "/cpfs01/shared/opendrivelab/GenAD_Datasets/YouTube",
        "path": "/cpfs01/shared/opendrivelab/opendrivelab_hdd/GenAD_Proj/eval/dataset_source/youtube/youtube_val_sub_<MODE>.json",
    }

    DATASET_IDX = {
        "nuscenes_val": NUSCENES_VAL,
        "youtube_val": YOUTUBE_VAL,
    }

    def __init__(self, initial, mode=None):
        assert isinstance(initial, dict)
        initial = EasyDict(initial)
        
        if "mode" in initial:
            assert initial.mode in ["image", "video"]
            if mode is None:
                mode = initial.mode
        if mode is None:
            mode = "image"
        
        if "dataset" in initial:
            self.update(self.DATASET_IDX[initial.dataset])
            self.path = self.path.replace("<MODE>", mode[0])
            self["freq"] = 12
            self["gen_startid"] = 0
        else:
            assert "root" in initial
            assert "path" in initial
            assert not isinstance(initial.path, dict), "`path` should be a list of path strings"
            self.update(initial)
            if "freq" not in self:
                self["freq"] = 12
            if "gen_startid" not in self:
                self["gen_startid"] = 1

        self._load_paths()
        self.pop("path")
        if (mode == 'image') and isinstance(self.source[0], list):
            frames = set()
            for video in self.source:
                for frm in video[self.gen_startid:]:
                    frames.add(frm)
            self.source = list(frames)
        assert ((mode == 'image') and isinstance(self.source[0], str)) or ((mode == 'video') and isinstance(self.source[0], list)), \
            "Please check if the metric and the dataset format (img/video) are matched"

    def _load_paths(self):
        source = json.load(open(self.path, "r"))
        if isinstance(source, dict):
            self.source = source["path"]
            for key in source:
                if key != "path":
                    self[key] = int(source[key])
        else:
            self.source = source


class Metric(object):
    def __init__(self, name):
        self.name = name
        print("Metric [{}] is initialized.".format(name))

    def update_gt(self, gt_paths, **kwargs):
        pass
    
    def forward(self, gen_paths, **kwargs):
        pass

    def __call__(self, gen_paths, **kwargs):
        return self.forward(gen_paths, **kwargs)



def open_and_resize(image_path):
        image = Image.open(image_path)
        ori_w, ori_h = image.size
        if ori_w / ori_h > TARGET_WIDTH / TARGET_HEIGHT:
            tmp_w = int(TARGET_WIDTH / TARGET_HEIGHT * ori_h)
            left = (ori_w - tmp_w) // 2
            right = (ori_w + tmp_w) // 2
            image = image.crop((left, 0, right, ori_h))
        image = image.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        return image