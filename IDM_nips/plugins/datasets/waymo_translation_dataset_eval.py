import json
import os

import numpy as np
from mmdet.datasets import DATASETS
from pyquaternion import Quaternion
import mmcv
from tqdm import tqdm

from .nuscenes_translation_dataset import NuScenesTranslationDataset


@DATASETS.register_module()
class WaymoTranslationDatasetEval(NuScenesTranslationDataset):
    CLASSES = ()

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 gen_image_root=None,
                 load_interval=1,
                 queue_length=8,
                 condition_frames=2,
                 test_mode=False,
                 sample_key="Sample",
                 final_cond_index=8,
                 **kwargs):
        self.load_interval = load_interval
        self.queue_length = queue_length
        self.condition_frames = condition_frames
        self.gen_image_root = gen_image_root

        self.sample_key = sample_key
        self.final_cond_index = final_cond_index  # * index of final conditional frame in the generation sequence
        self.index_to_gen_folder = dict()

        roots = gen_image_root
        if "GROUP" in roots.split('/')[-1]:
            roots = [
                os.path.join(roots, split) for split in os.listdir(roots) if not split.endswith(".json")
            ]

        if not isinstance(roots, list):
            roots = [roots]

        print(f"Roots: {roots}")

        for root in roots:
            
            for folder in tqdm(os.listdir(root)):

                collect_path = os.path.join(root, folder)
                # collect_path: folder 0, 1, 2, 3, ...

                if os.path.isfile(collect_path):
                    continue

                for name in os.listdir(collect_path):
                    gen_path = os.path.join(collect_path, name)
                    if os.path.isdir(gen_path) and name.startswith(self.sample_key):
                        
                        token = int(name.split("_")[-2])

                        self.index_to_gen_folder[token] = gen_path


        super(NuScenesTranslationDataset, self).__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            test_mode=test_mode,
            filter_empty_gt=False,
            **kwargs)

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file, file_format='json')
        data_infos = list(sorted(data['clips'], key=lambda e: e['token']))
        data_infos = data_infos[::self.load_interval]
        data_infos = self.get_planning_label(data_infos)
        return data_infos

    def get_planning_label(self, data_infos):
        planning_infos = []
        for index in range(len(data_infos)):
            cur_info = data_infos[index]
            token = cur_info['token']
            cmd_vista = cur_info['cmd_vista']

            if token not in self.index_to_gen_folder:
                print(f"Token {token} not found in generated folders, skipping...")
                continue

            gen_folder = self.index_to_gen_folder[token]
            gen_image_paths = os.listdir(gen_folder)
            gen_image_paths = [
                os.path.join(gen_folder, p) for p in gen_image_paths
            ]
            gen_image_paths = sorted(
                gen_image_paths, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )

            # TODO: Something is wrong with the idm, only see the future frames ???? How to estimate the position of first frame?
            gen_image_paths = gen_image_paths[self.final_cond_index: self.final_cond_index + 25]

            assert len(gen_image_paths) == 25

            gt_traj = cur_info['traj_fut']
            
            # * HARD Code, aligned with Vista
            # TODO: Remove Hard Code
            gt_traj = gt_traj[:4]

            planning_info = dict(
                # sample_idx=cur_info['token'],
                # scene_token=scene_token,
                token=token,  # * Used for sorting
                img_filename=gen_image_paths,
                gt_traj=np.array(gt_traj, dtype=np.float32),
                cmd_vista=cmd_vista,
            )
            planning_infos.append(planning_info)

        print("Total planning infos:", len(planning_infos))

        return planning_infos
