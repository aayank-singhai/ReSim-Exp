import copy
import json
import os

import cv2
import mmcv
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import Custom3DDataset
from pyquaternion import Quaternion
from tqdm import tqdm

cmd_vista_to_str = {
    0: "Right",
    1: "Left",
    2: "Stop",
    3: "Forward",
}

@DATASETS.register_module()
class WaymoTranslationDataset(Custom3DDataset):
    CLASSES = ()

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 load_interval=1,
                 queue_length=8,
                 condition_frames=2,
                 test_mode=False,
                 **kwargs):
        self.load_interval = load_interval
        self.queue_length = queue_length
        self.condition_frames = condition_frames

        super().__init__(
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
        for index in tqdm(range(len(data_infos))):
            cur_info = data_infos[index]
            token = cur_info['token']
            cmd_vista = cur_info['cmd_vista']

            image_paths = cur_info['img_seq']
            image_paths = image_paths[8: 33]
            image_paths = [os.path.join(self.data_root, p) for p in image_paths]

            assert len(image_paths) == 25

            gt_traj = cur_info['traj_fut']
            
            # * HARD Code, aligned with Vista
            # TODO: Remove Hard Code
            gt_traj = gt_traj[:4]

            planning_info = dict(
                token=token,  # * Used for sorting
                img_filename=image_paths,
                gt_traj=np.array(gt_traj, dtype=np.float32),
                cmd_vista=cmd_vista,
            )
            planning_infos.append(planning_info)

        return planning_infos
    
    def get_data_info(self, index):
        cur_info = self.data_infos[index]
        return copy.deepcopy(cur_info)

    def evaluate(self,
                 results,
                 metric='traj',
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 **kwargs):
        pred_trajs = []
        for i in range(len(results)):
            pred_trajs.append(results[i]['pred_traj'])

        gt_trajs = []
        for i in range(len(self.data_infos)):
            gt_trajs.append(self.data_infos[i]['gt_traj'])

        def ade_and_fde(pred_trajs, gt_trajs):
            pred_trajs = np.stack(pred_trajs, axis=0)[..., :2]  # (1500, 4, 2)
            gt_trajs = np.stack(gt_trajs, axis=0)[..., :2]      # (1500, 4, 2)

            ADE_1s = np.mean(
                np.sqrt(((pred_trajs[:, :2, :2] - gt_trajs[:, :2, :2]) ** 2).sum(axis=-1))
            )
            ADE_2s = np.mean(
                np.sqrt(((pred_trajs[:, :4, :2] - gt_trajs[:, :4, :2]) ** 2).sum(axis=-1))
            )
            ADE = np.mean(
                np.sqrt(((pred_trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2).sum(axis=-1))
            )
            FDE = np.mean(
                np.sqrt(((pred_trajs[:, -1, :2] - gt_trajs[:, -1, :2]) ** 2).sum(axis=-1))
            )

            return ADE_1s, ADE_2s, ADE, FDE

        ADE_1s, ADE_2s, ADE, FDE = ade_and_fde(pred_trajs, gt_trajs)
        

        pred_trajs_cmd = dict()
        gt_trajs_cmd = dict()
        
        for i in range(len(results)):
        
            pred = results[i]['pred_traj']
            gt   = self.data_infos[i]['gt_traj']

            data_info = self.data_infos[i]
            cmd_vista = data_info['cmd_vista']
            
            if cmd_vista not in pred_trajs_cmd:
                pred_trajs_cmd[cmd_vista] = [pred]
                gt_trajs_cmd[cmd_vista] = [gt]
            else:
                pred_trajs_cmd[cmd_vista].append(pred)
                gt_trajs_cmd[cmd_vista].append(gt)

        for cmd in pred_trajs_cmd.keys():
            _ADE_1s, _ADE_2s, _ADE, _FDE = ade_and_fde(pred_trajs_cmd[cmd], gt_trajs_cmd[cmd])
            print(f"\nCommand: {cmd}-{cmd_vista_to_str[cmd]}, ADE_1s: {_ADE_1s}, ADE_2s: {_ADE_2s}, ADE: {_ADE}, FDE: {_FDE}")

        if show:
            self.show(pred_trajs, gt_trajs, out_dir=out_dir, **kwargs)

        metrics = dict(
            ADE_1s=ADE_1s,
            ADE_2s=ADE_2s,
            ADE=ADE,
            FDE=FDE
        )
        return metrics

    def show(self, pred_trajs=None, gt_trajs=None, out_dir=None, show_num=100, **kwargs):

        def concat_8_images(images):
            h, w, c = images[0].shape
            canvas = np.zeros((h * 2, w * 4, c), dtype=np.uint8)
            canvas[:h, :w] = images[0]
            canvas[:h, w:2 * w] = images[1]
            canvas[:h, 2 * w:3 * w] = images[2]
            canvas[:h, 3 * w:] = images[3]
            canvas[h:, :w] = images[4]
            canvas[h:, w:2 * w] = images[5]
            canvas[h:, 2 * w:3 * w] = images[6]
            canvas[h:, 3 * w:] = images[7]
            return canvas

        def show_bev_results(gt_traj, pred_traj, map_size=[0, 50, -30, 30], scale=20):
            GT_COLOR = (0, 255, 0)
            PRED_COLOR = (0, 0, 255)
            canvas = np.zeros((int(scale * (map_size[1] - map_size[0])),
                               int(scale * (map_size[3] - map_size[2])), 3), dtype=np.uint8)
            # GT
            gt_canvas = np.zeros_like(canvas)
            draw_coor = (scale * (-gt_traj[:, :2] + np.array([map_size[1], map_size[3]]))).astype(int)
            gt_canvas = cv2.polylines(gt_canvas, [draw_coor[:, [1, 0]]], False, GT_COLOR, max(round(scale * 0.2), 1))
            for i in range(len(draw_coor)):
                gt_canvas = cv2.circle(gt_canvas, (draw_coor[i, 1], draw_coor[i, 0]), max(2, round(scale * 0.5)),
                                       GT_COLOR, -1)
            canvas = cv2.addWeighted(gt_canvas, 0.7, canvas, 1.0, 0)
            # PRED
            pred_canvas = np.zeros_like(canvas)
            draw_coor = (scale * (-pred_traj[:, :2] + np.array([map_size[1], map_size[3]]))).astype(int)
            pred_canvas = cv2.polylines(pred_canvas, [draw_coor[:, [1, 0]]], False, PRED_COLOR,
                                        max(round(scale * 0.2), 1))
            for i in range(len(draw_coor)):
                pred_canvas = cv2.circle(pred_canvas, (draw_coor[i, 1], draw_coor[i, 0]), max(2, round(scale * 0.5)),
                                         PRED_COLOR, -1)
            canvas = cv2.addWeighted(pred_canvas, 0.7, canvas, 1.0, 0)

            return canvas

        n_show = 0
        for idx in range(len(self.data_infos)):
            if idx % 6 != 0:
                continue
            if n_show >= show_num:
                break
            n_show += 1

            info = self.data_infos[idx]
            imgs = [cv2.imread(img) for img in info['img_filename']]
            image = concat_8_images(imgs)

            output_path = os.path.join(out_dir, 'show', str(info['timestamp']))
            os.makedirs(output_path, exist_ok=True)
            cv2.imwrite(f'{output_path}/image.jpg', image)

            traj_image = show_bev_results(
                np.concatenate([np.zeros((1, 2), dtype=np.float32), gt_trajs[idx]], axis=0),
                np.concatenate([np.zeros((1, 2), dtype=np.float32), pred_trajs[idx]], axis=0))
            cv2.imwrite(f'{output_path}/traj_pred.jpg', traj_image)
