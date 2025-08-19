import torch
from torch import nn

from mmdet.models import DETECTORS
from mmdet.models.detectors.base import BaseDetector
from mmdet.models.builder import build_loss

from .XVO import VOModelEncoder


@DETECTORS.register_module()
class VideoTranslatorFlow(BaseDetector):

    def __init__(self,
                 queue_length=8,
                 condition_frames=2,
                 feedforward_channels=512,
                 **kwargs):
        super(VideoTranslatorFlow, self).__init__()
        hard_code_queue_length = 25  # TODO: Check here
        self.xvo = VOModelEncoder()
        self.downsample = nn.Sequential(
            nn.Linear(20480, 64),
            nn.ReLU(inplace=True)
        )
        self.temporal_MLP = nn.Sequential(
            nn.Linear(64 * (hard_code_queue_length - 1), feedforward_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feedforward_channels, feedforward_channels // 2),
            nn.ReLU(inplace=True),
        )
        output_channel = 3 * (queue_length - condition_frames)
        self.translator = nn.Linear(feedforward_channels // 2, output_channel)
        self.loss = build_loss(dict(type='MSELoss', loss_weight=1.))

        for param in self.xvo.encoder.parameters():
            param.requires_grad = False

    def extract_feat(self, imgs):
        pass

    def forward_train(self, imgs, img_metas, gt_traj, **kwargs):
        assert imgs.dim() == 5
        B, N, C, H, W = imgs.size()
        assert N >= 2
        x_queue = []
        for i in range(N - 1):
            img_pair = imgs[:, i:i + 2]
            img_pair = img_pair.view(B, 2 * C, H, W)
            x = self.xvo(img_pair)  # B, 20480
            x_queue.append(x)
        x_queue = torch.stack(x_queue, dim=1)  # B, T, 20480

        # x_queue[:, 2:] = torch.zeros_like(x_queue[:, 2:])

        x_queue = self.downsample(x_queue)  # B, T, 128
        x_queue = x_queue.view(B, -1)
        x = self.temporal_MLP(x_queue)
        x = self.translator(x)

        pred = x.flatten()
        labels = torch.cat(gt_traj, 0).flatten()

        losses = dict()
        loss = self.loss(pred, labels)
        losses['loss_traj'] = loss
        return losses

    def forward_test(self, imgs, img_metas, **kwargs):
        assert imgs.dim() == 5
        B, N, C, H, W = imgs.size()
        assert N >= 2
        x_queue = []
        for i in range(N - 1):
            img_pair = imgs[:, i:i + 2]
            img_pair = img_pair.view(B, 2 * C, H, W)
            x = self.xvo(img_pair)
            x_queue.append(x)
        x_queue = torch.stack(x_queue, dim=1)  # B, T, 20480

        # x_queue[:, 2:] = torch.zeros_like(x_queue[:, 2:])

        x_queue = self.downsample(x_queue)  # B, T, 128
        x_queue = x_queue.view(B, -1)
        x = self.temporal_MLP(x_queue)
        x = self.translator(x)

        pred = x.cpu().numpy()
        results = []
        for i in range(pred.shape[0]):
            result = dict(pred_traj=pred[i].reshape(-1, 3))
            results.append(result)

        return results

    def simple_test(self, imgs, img_metas, **kwargs):
        pass

    def aug_test(self, imgs, img_metas, **kwargs):
        pass
