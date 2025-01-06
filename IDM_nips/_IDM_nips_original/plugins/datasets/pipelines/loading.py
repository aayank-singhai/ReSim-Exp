from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet3d.datasets.pipelines import DefaultFormatBundle


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        data = np.array(data)
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@PIPELINES.register_module()
class CustomLoadMultiViewImageFromFiles(object):

    def __init__(self, to_float32=False, color_type='bgr'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = [mmcv.imread(name, channel_order=self.color_type, backend='pillow') for name in filename]

        if self.to_float32:
            img = [_.astype(np.float32) for _ in img]
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = [img_.shape for img_ in img]
        results['ori_shape'] = [img_.shape for img_ in img]
        # Set initial values for default meta_keys
        results['pad_shape'] = [img_.shape for img_ in img]
        results['scale_factor'] = [1.0 for img_ in img]
        num_channels = 1 if len(img[0].shape) < 3 else img[0].shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class CustomDefaultFormatBundle(DefaultFormatBundle):

    def __call__(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        results = super(CustomDefaultFormatBundle, self).__call__(results)
        if 'gt_traj' in results:
            results['gt_traj'] = DC(to_tensor(results['gt_traj']))
        if 'gt_odo_pose' in results:
            results['gt_odo_pose'] = DC(to_tensor(results['gt_odo_pose']))
        if 'gt_rot_matrix' in results:
            results['gt_rot_matrix'] = DC(to_tensor(results['gt_rot_matrix']))

        return results


@PIPELINES.register_module()
class UseAutoEncoderData(object):

    def __init__(self, data_root, p_noisy=0.):
        self.data_root = data_root
        self.p_noisy = p_noisy

    def __call__(self, results):
        filename = []
        for fn in results['img_filename']:
            if "samples/" in fn:
                fn = fn.split('samples/')[-1]
                fn = f'{self.data_root}/{fn}'.replace('/CAM_FRONT/', '/CAM_FRONT_ae_clean/')
                if self.p_noisy == 1 or np.random.rand() < self.p_noisy:
                    fn = fn.replace('ae_clean', 'ae_noisy')
            else:
                fn = fn.split("sweeps/")[-1]
                fn = f'{self.data_root.replace("IDM_samples", "IDM_sweeps")}/{fn}'.replace('/CAM_FRONT/',
                                                                                           '/CAM_FRONT_ae_clean/')
                if self.p_noisy == 1 or np.random.rand() < self.p_noisy:
                    fn = fn.replace('ae_clean', 'ae_noisy')
            filename.append(fn)
        results['img_filename'] = filename
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(p_noisy={self.p_noisy})'
        return repr_str
