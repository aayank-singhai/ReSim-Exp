# some global variables and default dataset settings

import os
import json
from PIL import Image

# import pynvml

from utils.easydict import EasyDict

I3D_PATH = "/cpfs01/shared/opendrivelab/opendrivelab_hdd/qiuyihang/models/i3d/i3d_pretrained_400.pt"
MAX_BATCH = 8
TARGET_RESOLUTION = (224, 224)
MIN_I3D_TIME = 9
TARGET_WIDTH = 448
TARGET_HEIGHT = 256

VID_METRICS = ["fvd", "kvd", "clipsim", "cmd_fvd"]
IMG_METRICS = ["fid", "is"]
PAIRED_DATASET_SUBSET_FLAG = "<SUBSET>"


class CustomizedPairedDataSource(EasyDict):
    # expected folder structure
    # <root>
    # |-- <splitxx>
    # |   |-- <real> 
    # |   |   |-- <images>
    # |   |   |   |-- <xxxxxx_[CLIP ID]_[FRAME INDEX]>.<FORMAT>
    # |   |   |-- other folders
    # |   |   |-- ...
    # |   |-- <virtual> 
    # |   |   |-- <images>
    # |   |   |-- other folders
    # |   |   |-- ...
    
    SUBSET_FLAG = PAIRED_DATASET_SUBSET_FLAG
    DEFAULT = {
        "format": "png",
        "gt_key": "real",
        "gen_key": "virtual",
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
        self._sweep_for_all("backup_root" in initial.keys())

        print("You are loading paired dataset from [{}]".format(self.root))
        sample = self.source[0]
        if isinstance(sample, list):
            sample = sample[0]
        print("A GT sample is as follows: [{}]".format(sample.replace(self.SUBSET_FLAG, self.gt_key)))
        print("#(samples): [{}]".format(len(self.source)))

        if self.supp is not None:
            print("GT supplement is loaded from [{}]".format(self.supp))
        

    def _sweep_for_all(self, has_backup=False):
        root = self.backup_root if has_backup else self.root
        for split in os.listdir(root):
            folders = os.listdir(os.path.join(root, split))
            assert "real" in folders, \
                "no `real` folder found in {}. Please check your dataset.".format(os.path.join(root, split))
            
            collect_path = os.path.join(split, self.SUBSET_FLAG, "images")
            file_list = os.listdir(os.path.join(root, collect_path.replace(self.SUBSET_FLAG, self.gt_key)))
            for sample in file_list:
                if sample.endswith(self.format):
                    break
            
            sample = sample.split(".")[0]
            clip_id_width = len(sample.split("_")[-2])
            frm_id_width = len(sample.split("_")[-1])
            prefix = "_".join(sample.split("_")[:-2])
            clip_count = len(file_list) // self.max_frameid

            for i in range(clip_count):
                self.source.append([])
                for j in range(self.gen_startid, self.max_frameid):
                    self.source[-1].append( os.path.join(collect_path, \
                                "{}_{}_{}.{}".format(prefix, str(i).zfill(clip_id_width), 
                                                        str(j).zfill(frm_id_width), self.format)) )
            
        if self.recollect:
            tmp = []
            for x in self.source:
                tmp.extend(x)
            self.source = tmp

    def switch_to_subset(self, subset):
        self.paired_subset = subset

    def gt(self):
        self.paired_subset = self.gt_key
    
    def gen(self):
        self.paired_subset = self.gen_key

    def get_index(self, filename):
        split_id = int(filename.split("/")[-4].split("split")[-1])
        scene_id = int(filename.split("/")[-1].split("_")[1])
        return split_id * self.split_len + scene_id




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



def gpu_state():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print("GPU memory: {}/{}".format(meminfo.used / 1024**2, meminfo.total / 1024**2))
    print("Free memory: {}".format(meminfo.free / 1024**2))
    print("--------------------------------------")



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