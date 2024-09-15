from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig
import math

from ...modules.diffusionmodules.sampling import VideoDDIMSampler, VPSDEDPMPP2MSampler
from ...util import append_dims, instantiate_from_config
from ...modules.autoencoding.lpips.loss.lpips import LPIPS

# import rearrange
from einops import rearrange
import random
from sat import mpu


class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config,
        type="l2",
        offset_noise_level=0.0,
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
    ):
        super().__init__()

        assert type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)

        self.type = type
        self.offset_noise_level = offset_noise_level

        if type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}

        sigmas = self.sigma_sampler(input.shape[0]).to(input.device)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            noise = (
                noise + append_dims(torch.randn(input.shape[0]).to(input.device), input.ndim) * self.offset_noise_level
            )
            noise = noise.to(input.dtype)
        noised_input = input.float() + noise * append_dims(sigmas, input.ndim)
        model_output = denoiser(network, noised_input, sigmas, cond, **additional_model_inputs)
        w = append_dims(denoiser.w(sigmas), input.ndim)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss


class VideoDiffusionLoss(StandardDiffusionLoss):
    def __init__(self, block_scale=None, block_size=None, min_snr_value=None, fixed_frames=0, cond_inds=None, apply_cond_aug=False,
                **kwargs):
        self.fixed_frames = fixed_frames  # TODO: Remove this?
        self.block_scale = block_scale
        self.block_size = block_size
        self.min_snr_value = min_snr_value
        super().__init__(**kwargs)
        self.cond_inds = cond_inds  # [0, 1, 2] video latents <--> corresponding [0, 8] (n=9) video frames
        self.apply_cond_aug = apply_cond_aug  # apply aug on conditioning frames

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}

        alphas_cumprod_sqrt, idx = self.sigma_sampler(input.shape[0], return_idx=True)
        alphas_cumprod_sqrt = alphas_cumprod_sqrt.to(input.device)  # a float
        idx = idx.to(input.device)  # t

        # print(f"idx:{idx}, alpha_t:{alphas_cumprod_sqrt}")
        # idx:tensor([26], device='cuda:0'), alpha_t:tensor([0.9631], device='cuda:0')
        # idx:tensor([335], device='cuda:0'), alpha_t:tensor([0.5044], device='cuda:0')
        # idx:tensor([510], device='cuda:1'), alpha_t:tensor([0.2985], device='cuda:1')

        cond_inds = self.cond_inds
        
        if cond_inds is not None:
            cond_mask = torch.zeros(input.shape).to(input.device) # [1, 13, 16, 64, 112]
            cond_mask[:, cond_inds] = 1

        noise = torch.randn_like(input)

        # broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        global_rank = torch.distributed.get_rank() // mp_size
        src = global_rank * mp_size
        torch.distributed.broadcast(idx, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(noise, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(alphas_cumprod_sqrt, src=src, group=mpu.get_model_parallel_group())

        additional_model_inputs["idx"] = idx  # TODO: Check idx

        if self.offset_noise_level > 0.0:
            noise = (
                noise + append_dims(torch.randn(input.shape[0]).to(input.device), input.ndim) * self.offset_noise_level
            )

        # x_t
        noised_input = input.float() * append_dims(alphas_cumprod_sqrt, input.ndim) + noise * append_dims(
            (1 - alphas_cumprod_sqrt**2) ** 0.5, input.ndim
        )
        # noised_input: torch.Size([1, 13, 16, 64, 112])
        if cond_inds is not None:
            additional_model_inputs['cond_inds'] = cond_inds

            aug_input = input.clone()
            if self.apply_cond_aug:
                # * Apply augmentation on conditioning frames.

                # TODO: Noiser distribution with Chunk indicator? e.g., 0-100: 0, 100-200: 100, 200-300: 200
                # TODO: Use forward diffusion process on conditioning frames, instead of simple addition.
                log_cond_aug_dist = torch.distributions.Normal(-3.0, 0.5)  # * Following SVD
                log_cond_aug = log_cond_aug_dist.sample()
                cond_aug = torch.exp(log_cond_aug)
                aug_input = aug_input + cond_aug * torch.randn_like(aug_input)

            # * Replace noised latents with conditioning ones.
            noised_input = aug_input.float() * cond_mask + \
                           noised_input * (1 - cond_mask)

        model_output = denoiser(network, noised_input, alphas_cumprod_sqrt, cond, **additional_model_inputs)
        # model_output: torch.Size([1, 13, 16, 64, 112]), b t c h w
        w = append_dims(1 / (1 - alphas_cumprod_sqrt**2), input.ndim)  # v-pred  torch.Size([1, 1, 1, 1, 1])
        b, t = model_output.shape[:2]

        if cond_inds is not None:
            model_output = model_output[~cond_mask.bool()].reshape(b, -1, *model_output.shape[2:])
            input = input[~cond_mask.bool()].reshape(b, -1, *input.shape[2:])

        # TODO: Try this, min_snr
        if self.min_snr_value is not None:
            # w = min(w, self.min_snr_value)  # Minor issue here: w might not be a tensor
            w = torch.where(w > self.min_snr_value, self.min_snr_value, w)

        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss


def get_3d_position_ids(frame_len, h, w):
    i = torch.arange(frame_len).view(frame_len, 1, 1).expand(frame_len, h, w)
    j = torch.arange(h).view(1, h, 1).expand(frame_len, h, w)
    k = torch.arange(w).view(1, 1, w).expand(frame_len, h, w)
    position_ids = torch.stack([i, j, k], dim=-1).reshape(-1, 3)
    return position_ids
