import json
import os

import numpy as np
from mmdet.datasets import DATASETS
from pyquaternion import Quaternion

from .nuscenes_translation_dataset import NuScenesTranslationDataset


@DATASETS.register_module()
class NuScenesTranslationDatasetEval(NuScenesTranslationDataset):
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
                 **kwargs):
        self.load_interval = load_interval
        self.queue_length = queue_length
        self.condition_frames = condition_frames
        self.gen_image_root = gen_image_root

        nusc_file = "/cpfs01/user/gaoshenyuan/nuScenes_svd_val.json"
        with open(nusc_file, "r") as nusc_json:
            nusc_samples = json.load(nusc_json)[::10]

        self.index_mapping = dict()
        for idx, sample in enumerate(nusc_samples):
            self.index_mapping[sample["frames"][0].split("/")[-1]] = idx

        super(NuScenesTranslationDataset, self).__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            test_mode=test_mode,
            filter_empty_gt=False,
            **kwargs)

    def translate_split_info(self, abs_index):
        split_index = abs_index // 59
        in_split_index = abs_index % 59
        return split_index, in_split_index

    def get_planning_label(self, data_infos):
        planning_infos = []
        for index in range(len(data_infos)):
            cur_info = data_infos[index]

            front_img_fn = os.path.basename(cur_info['cams']['CAM_FRONT']['data_path'])

            if front_img_fn not in self.index_mapping.keys():
                continue

            sample_index = self.index_mapping[front_img_fn]
            split_index, in_split_index = self.translate_split_info(sample_index)

            scene_token = cur_info['scene_token']
            timestamp = cur_info['timestamp']
            e2g_translation = np.array(cur_info['ego2global_translation'])
            e2g_rotation = np.array(cur_info['ego2global_rotation'])
            e2g_rotation = Quaternion(e2g_rotation).rotation_matrix
            e2g_matrix = np.eye(4)
            e2g_matrix[:3, :3] = e2g_rotation
            e2g_matrix[:3, 3] = e2g_translation
            g2e_matrix = np.linalg.inv(e2g_matrix)

            # image_paths = []
            traj = np.zeros((self.queue_length, 3), dtype=np.float32)
            traj_mask = np.zeros((self.queue_length, 1), dtype=bool)

            indices = list(range(index, index + self.queue_length))

            for idx, frame_id in enumerate(indices):
                if frame_id >= len(data_infos) or frame_id < 0:
                    break
                info = data_infos[frame_id]
                if info['scene_token'] != scene_token:
                    break

                traj_mask[idx] = True
                e2g_t = np.array(info['ego2global_translation'])
                traj[idx] = e2g_t

                # image_path = info['cams']['CAM_FRONT']['data_path']
                # # abs path to relative path
                # image_path = os.path.join(self.data_root, image_path)
                # image_paths.append(image_path)

            # keyframe_indices = [0, 6, 12, 18, 24]
            # gen_image_paths = [
            #     f'{self.gen_image_root}/split{split_index:02}/virtual/images/NUSCENES_{in_split_index:06}_{key_i:04}.png'
            #     for key_i in keyframe_indices]
            gen_image_paths = [
                f'{self.gen_image_root}/split{split_index:02}/virtual/images/NUSCENES_{in_split_index:06}_{key_i:04}.png'
                for key_i in range(25)]

            traj_xyz1 = np.concatenate([traj, np.ones((self.queue_length, 1))], axis=1)
            traj_xyz1 = np.matmul(traj_xyz1, g2e_matrix.T)
            traj = traj_xyz1[self.condition_frames:, :3]

            if traj_mask.sum() < self.queue_length:
                continue

            planning_info = dict(
                sample_idx=cur_info['token'],
                scene_token=scene_token,
                timestamp=timestamp,
                img_filename=gen_image_paths,
                gt_traj=traj.astype(np.float32)
            )
            planning_infos.append(planning_info)

        return planning_infos
