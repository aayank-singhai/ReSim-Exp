import math
from typing import Callable, Dict, Iterable

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer

OptimizerCallable = Callable[[Iterable], Optimizer]

import timm
from einops import rearrange
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class RewardModel(LightningModule):
    def __init__(
            self,
            model_dim: int = 256,
            optimizer: OptimizerCallable = AdamW
    ) -> None:
        super(RewardModel, self).__init__()
        self.dino_encoder = torch.hub.load("/cpfs01/shared/opendrivelab/opendrivelab_hdd/gaoshenyuan/dinov2",
                                           "dinov2_vitb14_reg", source="local", pretrained=False)
        self.dino_encoder.load_state_dict(
            torch.load("/cpfs01/shared/opendrivelab/opendrivelab_hdd/gaoshenyuan/dinov2_vitb14_reg4_pretrain.pth",
                       map_location="cpu"))
        del self.dino_encoder.head
        self.dino_encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            self.dino_encoder.pos_embed.data, [16, 16]
        )
        self.dino_encoder.head = nn.Identity()
        self.dino_encoder.eval()

        self.model_dim = model_dim

        self.downsample = DownsampleBlock(self.dino_encoder.embed_dim, model_dim)
        self.pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.estimator = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, model_dim // 4),
            nn.GELU(),
            nn.Linear(model_dim // 4, 1),
            nn.Sigmoid()
        )
        self.loss = nn.MSELoss()

        self.optimizer = optimizer
        self.save_hyperparameters()

    def shared_step(self, batch: Dict) -> Tensor:
        videos = batch["videos"]
        B, T = videos.shape[:2]

        with torch.no_grad():
            # Encode the videos using DINOv2
            dino_inputs = rearrange(videos, "b t c h w -> (b t) c h w")
            dino_inputs = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(dino_inputs)
            dino_z = self.dino_encoder.forward_features(dino_inputs)["x_norm_patchtokens"]
            edge = int(math.sqrt(dino_z.shape[1]))
            dino_z = rearrange(dino_z, "(b t) (h w) e -> b e t h w", b=B, t=T, h=edge, w=edge).detach()

        dino_z = self.downsample(dino_z)
        pooled_z = self.pooling(dino_z).squeeze()
        reward_estimation = self.estimator(pooled_z)

        # Compute loss
        loss = self.loss(reward_estimation, batch["rewards"])
        return loss

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the training loss
        batch_loss = self.shared_step(batch)

        # Log the training loss
        self.log_dict(
            {**{"train_loss": batch_loss}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False
        )
        return batch_loss

    @torch.no_grad()
    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the validation loss
        batch_loss = self.shared_step(batch)

        # Log the validation loss
        self.log_dict(
            {**{"val_loss": batch_loss}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )
        return batch_loss

    @torch.no_grad()
    def test_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the test loss
        batch_loss = self.shared_step(batch)

        # Log the test loss
        self.log_dict(
            {**{"test_loss": batch_loss}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )
        return batch_loss

    def configure_optimizers(self) -> Optimizer:
        return self.optimizer(self.parameters())
