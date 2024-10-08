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
    def __init__(self, block_scale=None, block_size=None, min_snr_value=None, fixed_frames=0, 
                 cond_inds=None,
                 cond_inds_prob=None,
                 apply_cond_aug=None,
                 max_aug_t=700,  # * For V2
                 contained_aug_t=False,
                 exclude_cond_from_loss=True,
                **kwargs):
        self.fixed_frames = fixed_frames  # TODO: Remove this?
        self.block_scale = block_scale
        self.block_size = block_size
        self.min_snr_value = min_snr_value
        super().__init__(**kwargs)
        
        self.cond_inds = cond_inds # Nested list. e.g.,  [0, 1, 2] should be [ [0, 1, 2] ]

        self.cond_inds_prob = cond_inds_prob  # probability of different conditioning schemes, None for uniform distribution
        assert cond_inds_prob is None or len(cond_inds_prob) == len(cond_inds), f"Invalid cond_inds_prob: {cond_inds_prob}"

        self.apply_cond_aug = apply_cond_aug  # apply aug on conditioning frames
        assert self.apply_cond_aug in [None, 'V1', 'V2'], f"Invalid apply_cond_aug: {self.apply_cond_aug}"

        self.max_aug_t = max_aug_t  # * For V2
        self.contained_aug_t = contained_aug_t  # contain the aug_t to be no greater than t (on prediction frames)
        self.exclude_cond_from_loss = exclude_cond_from_loss

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}

        alphas_cumprod_sqrt, idx = self.sigma_sampler(input.shape[0], return_idx=True)
        alphas_cumprod_sqrt = alphas_cumprod_sqrt.to(input.device)  # a float
        idx = idx.to(input.device)  # indicating t. shape: [bs]. eg: [845, 10]

        # print(f"idx:{idx}, alpha_t:{alphas_cumprod_sqrt}")
        # idx:tensor([26], device='cuda:0'), alpha_t:tensor([0.9631], device='cuda:0')
        # idx:tensor([335], device='cuda:0'), alpha_t:tensor([0.5044], device='cuda:0')
        # idx:tensor([510], device='cuda:1'), alpha_t:tensor([0.2985], device='cuda:1')

        cond_inds = self.cond_inds
        if cond_inds is not None:
            cond_inds = random.choices(cond_inds, weights=self.cond_inds_prob, k=1)[0]  # randomly choose a conditioning scheme
            cond_inds = list(cond_inds)  # convert from omegaconf to normal list
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

        # Add traj here.
        additional_model_inputs["with_traj"] = batch["with_traj"]
        additional_model_inputs["fut_traj"] = batch["fut_traj"]

        # TODO: Use transformer to encode the traj.

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
            if self.apply_cond_aug == 'V1':
                # * Apply augmentation on conditioning frames.
                # * Weakness: mild augmentation, condition frames are not noisy enough.
                log_cond_aug_dist = torch.distributions.Normal(-3.0, 0.5)  # * Following SVD
                log_cond_aug = log_cond_aug_dist.sample()
                cond_aug = torch.exp(log_cond_aug)
                aug_input = aug_input + cond_aug * torch.randn_like(aug_input)
            
            elif self.apply_cond_aug == 'V2':
                # * Improved, noiser aug, diffusion forward process on conditioning frames.
                if self.contained_aug_t:
                    lb = idx.new_zeros(idx.shape)
                    ub = torch.where(idx > self.max_aug_t, self.max_aug_t, idx)
                    aug_t = torch.tensor([torch.randint(l, h+1, (1,)).item() for l, h in zip(lb, ub)])
                else:
                    aug_t = torch.randint(0, self.max_aug_t, (input.shape[0],))  # [0, max_aug_t(excluded )

                aug_t_chunk = aug_t // 100 * 100  # 0-100: 0, 100-200: 100, 200-300: 200
                additional_model_inputs['aug_t_chunk'] = aug_t_chunk.to(input.device)
                # * Inference: set 0
                
                aug_alphas_cumprod_sqrt = self.sigma_sampler.sigmas[aug_t]  # Checked, correct order self.sigma_sampler.sigmas[10]: tensor(0.9853)
                aug_alphas_cumprod_sqrt = aug_alphas_cumprod_sqrt.to(input.device)
                aug_noise = torch.randn_like(input)

                # aug_x for conditioning frames
                aug_input = input.float() * append_dims(aug_alphas_cumprod_sqrt, input.ndim) + aug_noise * append_dims(
                    (1 - aug_alphas_cumprod_sqrt**2) ** 0.5, input.ndim
                )

            # * Replace noised latents with conditioning ones.
            noised_input = aug_input.float() * cond_mask + \
                           noised_input * (1 - cond_mask)

        model_output = denoiser(network, noised_input, alphas_cumprod_sqrt, cond, **additional_model_inputs)  # go to DiscreteDenoiser
        # model_output: torch.Size([1, 13, 16, 64, 112]), b t c h w
        w = append_dims(1 / (1 - alphas_cumprod_sqrt**2), input.ndim)  # v-pred  torch.Size([1, 1, 1, 1, 1])
        b, t = model_output.shape[:2]

        # * Exclude the condition frames from loss
        if cond_inds is not None and self.exclude_cond_from_loss:
            model_output = model_output[~cond_mask.bool()].reshape(b, -1, *model_output.shape[2:])
            input = input[~cond_mask.bool()].reshape(b, -1, *input.shape[2:])

        # Tried, not working: min_snr
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
