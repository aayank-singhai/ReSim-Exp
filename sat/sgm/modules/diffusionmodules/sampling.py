"""
Partially ported from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
"""

from typing import Dict, Union

import torch
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm

from ...modules.diffusionmodules.sampling_utils import (
    get_ancestral_step,
    linear_multistep_coeff,
    to_d,
    to_neg_log_sigma,
    to_sigma,
)
from ...util import append_dims, default, instantiate_from_config
from ...util import SeededNoise

from .guiders import DynamicCFG, TriangleCFG

DEFAULT_GUIDER = {"target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"}


class BaseDiffusionSampler:
    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig, OmegaConf],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
        verbose: bool = False,
        device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)
        self.guider = instantiate_from_config(
            default(
                guider_config,
                DEFAULT_GUIDER,
            )
        )
        self.verbose = verbose
        self.device = device

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(self.num_steps if num_steps is None else num_steps, device=self.device)
        uc = default(uc, cond)

        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]]).float()

        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x, denoiser, sigma, cond, uc):
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc))
        denoised = self.guider(denoised, sigma)
        return denoised

    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps",
            )
        return sigma_generator


class SingleStepDiffusionSampler(BaseDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        return x + dt * d


class EDMSampler(SingleStepDiffusionSampler):
    def __init__(self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0.0):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        denoised = self.denoise(x, denoiser, sigma_hat, cond, uc)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(euler_step, x, d, dt, next_sigma, denoiser, cond, uc)
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        for i in self.get_sigma_gen(num_sigmas):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1) if self.s_tmin <= sigmas[i] <= self.s_tmax else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma,
            )

        return x


class DDIMSampler(SingleStepDiffusionSampler):
    def __init__(self, s_noise=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.s_noise = s_noise

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, s_noise=0.0):
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        d = to_d(x, sigma, denoised)
        dt = append_dims(next_sigma * (1 - s_noise**2) ** 0.5 - sigma, x.ndim)

        euler_step = x + dt * d + s_noise * append_dims(next_sigma, x.ndim) * torch.randn_like(x)

        x = self.possible_correction_step(euler_step, x, d, dt, next_sigma, denoiser, cond, uc)
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        for i in self.get_sigma_gen(num_sigmas):
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                self.s_noise,
            )

        return x


class AncestralSampler(SingleStepDiffusionSampler):
    def __init__(self, eta=1.0, s_noise=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eta = eta
        self.s_noise = s_noise
        self.noise_sampler = lambda x: torch.randn_like(x)

    def ancestral_euler_step(self, x, denoised, sigma, sigma_down):
        d = to_d(x, sigma, denoised)
        dt = append_dims(sigma_down - sigma, x.ndim)

        return self.euler_step(x, d, dt)

    def ancestral_step(self, x, sigma, next_sigma, sigma_up):
        x = torch.where(
            append_dims(next_sigma, x.ndim) > 0.0,
            x + self.noise_sampler(x) * self.s_noise * append_dims(sigma_up, x.ndim),
            x,
        )
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        for i in self.get_sigma_gen(num_sigmas):
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
            )

        return x


class LinearMultistepSampler(BaseDiffusionSampler):
    def __init__(
        self,
        order=4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.order = order

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        ds = []
        sigmas_cpu = sigmas.detach().cpu().numpy()
        for i in self.get_sigma_gen(num_sigmas):
            sigma = s_in * sigmas[i]
            denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc), **kwargs)
            denoised = self.guider(denoised, sigma)
            d = to_d(x, sigma, denoised)
            ds.append(d)
            if len(ds) > self.order:
                ds.pop(0)
            cur_order = min(i + 1, self.order)
            coeffs = [linear_multistep_coeff(cur_order, sigmas_cpu, i, j) for j in range(cur_order)]
            x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))

        return x


class EulerEDMSampler(EDMSampler):
    def possible_correction_step(self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc):
        return euler_step


class HeunEDMSampler(EDMSampler):
    def possible_correction_step(self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc):
        if torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            return euler_step
        else:
            denoised = self.denoise(euler_step, denoiser, next_sigma, cond, uc)
            d_new = to_d(euler_step, next_sigma, denoised)
            d_prime = (d + d_new) / 2.0

            # apply correction if noise level is not 0
            x = torch.where(append_dims(next_sigma, x.ndim) > 0.0, x + d_prime * dt, euler_step)
            return x


class EulerAncestralSampler(AncestralSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x = self.ancestral_euler_step(x, denoised, sigma, sigma_down)
        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)

        return x


class DPMPP2SAncestralSampler(AncestralSampler):
    def get_variables(self, sigma, sigma_down):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, sigma_down)]
        h = t_next - t
        s = t + 0.5 * h
        return h, s, t, t_next

    def get_mult(self, h, s, t, t_next):
        mult1 = to_sigma(s) / to_sigma(t)
        mult2 = (-0.5 * h).expm1()
        mult3 = to_sigma(t_next) / to_sigma(t)
        mult4 = (-h).expm1()

        return mult1, mult2, mult3, mult4

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, **kwargs):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x_euler = self.ancestral_euler_step(x, denoised, sigma, sigma_down)

        if torch.sum(sigma_down) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            x = x_euler
        else:
            h, s, t, t_next = self.get_variables(sigma, sigma_down)
            mult = [append_dims(mult, x.ndim) for mult in self.get_mult(h, s, t, t_next)]

            x2 = mult[0] * x - mult[1] * denoised
            denoised2 = self.denoise(x2, denoiser, to_sigma(s), cond, uc)
            x_dpmpp2s = mult[2] * x - mult[3] * denoised2

            # apply correction if noise level is not 0
            x = torch.where(append_dims(sigma_down, x.ndim) > 0.0, x_dpmpp2s, x_euler)

        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)
        return x


class DPMPP2MSampler(BaseDiffusionSampler):
    def get_variables(self, sigma, next_sigma, previous_sigma=None):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, next_sigma)]
        h = t_next - t

        if previous_sigma is not None:
            h_last = t - to_neg_log_sigma(previous_sigma)
            r = h_last / h
            return h, r, t, t_next
        else:
            return h, None, t, t_next

    def get_mult(self, h, r, t, t_next, previous_sigma):
        mult1 = to_sigma(t_next) / to_sigma(t)
        mult2 = (-h).expm1()

        if previous_sigma is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def sampler_step(
        self,
        old_denoised,
        previous_sigma,
        sigma,
        next_sigma,
        denoiser,
        x,
        cond,
        uc=None,
    ):
        denoised = self.denoise(x, denoiser, sigma, cond, uc)

        h, r, t, t_next = self.get_variables(sigma, next_sigma, previous_sigma)
        mult = [append_dims(mult, x.ndim) for mult in self.get_mult(h, r, t, t_next, previous_sigma)]

        x_standard = mult[0] * x - mult[1] * denoised
        if old_denoised is None or torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return x_standard, denoised
        else:
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d

            # apply correction if noise level is not 0 and not first step
            x = torch.where(append_dims(next_sigma, x.ndim) > 0.0, x_advanced, x_standard)

        return x, denoised

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        old_denoised = None
        for i in self.get_sigma_gen(num_sigmas):
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * sigmas[i - 1],
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc=uc,
            )

        return x


class SDEDPMPP2MSampler(BaseDiffusionSampler):
    def get_variables(self, sigma, next_sigma, previous_sigma=None):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, next_sigma)]
        h = t_next - t

        if previous_sigma is not None:
            h_last = t - to_neg_log_sigma(previous_sigma)
            r = h_last / h
            return h, r, t, t_next
        else:
            return h, None, t, t_next

    def get_mult(self, h, r, t, t_next, previous_sigma):
        mult1 = to_sigma(t_next) / to_sigma(t) * (-h).exp()
        mult2 = (-2 * h).expm1()

        if previous_sigma is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def sampler_step(
        self,
        old_denoised,
        previous_sigma,
        sigma,
        next_sigma,
        denoiser,
        x,
        cond,
        uc=None,
    ):
        denoised = self.denoise(x, denoiser, sigma, cond, uc)

        h, r, t, t_next = self.get_variables(sigma, next_sigma, previous_sigma)
        mult = [append_dims(mult, x.ndim) for mult in self.get_mult(h, r, t, t_next, previous_sigma)]
        mult_noise = append_dims(next_sigma * (1 - (-2 * h).exp()) ** 0.5, x.ndim)

        x_standard = mult[0] * x - mult[1] * denoised + mult_noise * torch.randn_like(x)
        if old_denoised is None or torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return x_standard, denoised
        else:
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d + mult_noise * torch.randn_like(x)

            # apply correction if noise level is not 0 and not first step
            x = torch.where(append_dims(next_sigma, x.ndim) > 0.0, x_advanced, x_standard)

        return x, denoised

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, scale=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        old_denoised = None
        for i in self.get_sigma_gen(num_sigmas):
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * sigmas[i - 1],
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc=uc,
            )

        return x


class SdeditEDMSampler(EulerEDMSampler):
    def __init__(self, edit_ratio=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.edit_ratio = edit_ratio

    def __call__(self, denoiser, image, randn, cond, uc=None, num_steps=None, edit_ratio=None):
        randn_unit = randn.clone()
        randn, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(randn, cond, uc, num_steps)

        if num_steps is None:
            num_steps = self.num_steps
        if edit_ratio is None:
            edit_ratio = self.edit_ratio
        x = None

        for i in self.get_sigma_gen(num_sigmas):
            if i / num_steps < edit_ratio:
                continue
            if x is None:
                x = image + randn_unit * append_dims(s_in * sigmas[i], len(randn_unit.shape))

            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1) if self.s_tmin <= sigmas[i] <= self.s_tmax else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma,
            )

        return x


class VideoDDIMSampler(BaseDiffusionSampler):
    def __init__(self, fixed_frames=None, sdedit=False, cond_inds_sampling=None, apply_cond_aug=None, 
                 apply_cond_aug_chunk_inference = 'zero',
                 fix_timestep_bug = False,
                 directly_use_idx_as_timestep = False, # sampling_timestep for guider wrt. t: [25, 24, ...., 0]
                 **kwargs):
        super().__init__(**kwargs)
        self.fixed_frames = fixed_frames
        self.sdedit = sdedit
        self.cond_inds = cond_inds_sampling
        self.apply_cond_aug = apply_cond_aug
        assert self.apply_cond_aug in [None, 'V1', 'V2'], f"Invalid apply_cond_aug: {self.apply_cond_aug}"
        self.fix_timestep_bug = fix_timestep_bug
        self.directly_use_idx_as_timestep = directly_use_idx_as_timestep

        self.apply_cond_aug_chunk_inference = apply_cond_aug_chunk_inference
        assert isinstance(self.apply_cond_aug_chunk_inference, int) or \
                self.apply_cond_aug_chunk_inference in ['v1', 'zero', 'min', 'dynamic']
        # * v1 for randomly sampled gaussian noise

        # * Fixed value
        # * - chunk 0 --> timestep: 50
        # * - chunk 1 --> timestep: 150
        # * - chunk x --> timestep: x * 100 + 50
        # * Zero
        # * - no effect
        # * Min
        # * - chunk 0 --> timestep: 0
        # * Dynamic
        # * - gradually increase the timestep chunk from 0 -> 7
        # * - then chunk x --> timestep: x * 100 + 50

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        alpha_cumprod_sqrt, timesteps = self.discretization(
            self.num_steps if num_steps is None else num_steps,
            device=self.device,
            return_idx=True,
            do_append_zero=False,
        )
        alpha_cumprod_sqrt = torch.cat([alpha_cumprod_sqrt, alpha_cumprod_sqrt.new_ones([1])])
        timesteps = torch.cat([torch.tensor(list(timesteps)).new_zeros([1]) - 1, torch.tensor(list(timesteps))])

        uc = default(uc, cond)

        num_sigmas = len(alpha_cumprod_sqrt)

        s_in = x.new_ones([x.shape[0]])

        return x, s_in, alpha_cumprod_sqrt, num_sigmas, cond, uc, timesteps

    def denoise(self, x, denoiser, alpha_cumprod_sqrt, cond, uc, timestep=None, idx=None, scale=None, scale_emb=None, **additional_model_inputs):

        additional_model_inputs["is_sampling"] = True  # * Used to indicate sampling

        if self.cond_inds is not None:
            additional_model_inputs['cond_inds'] = self.cond_inds
        
        aug_t_chunk_sampling = additional_model_inputs.get('aug_t_chunk', 0)
        # import pdb; pdb.set_trace()
        # * scale: None
        # * We don't need to pass the scale as it will be initialized in the guider

        if not isinstance(scale, torch.Tensor) and scale == 1:
            additional_model_inputs["idx"] = x.new_ones([x.shape[0]]) * timestep  # TODO: Make sure this timestep is right

            # TODO: Check this?? A bug??? the conflict between apply_cond_aug and apply_cond_aug_chunk_inference
            if self.apply_cond_aug == 'V2':
                additional_model_inputs["aug_t_chunk"] = x.new_ones([x.shape[0]]) * aug_t_chunk_sampling   # TODO: Make sure this aug_t_chunk_sampling is right
            if scale_emb is not None:
                additional_model_inputs["scale_emb"] = scale_emb
            denoised = denoiser(x, alpha_cumprod_sqrt, cond, **additional_model_inputs).to(torch.float32)
        else:
            additional_model_inputs["idx"] = torch.cat([x.new_ones([x.shape[0]]) * timestep] * 2)
            if self.apply_cond_aug == 'V2':
                additional_model_inputs["aug_t_chunk"] = torch.cat([x.new_ones([x.shape[0]])] * 2) * aug_t_chunk_sampling
            denoised = denoiser(
                *self.guider.prepare_inputs(x, alpha_cumprod_sqrt, cond, uc), **additional_model_inputs
            ).to(torch.float32)
            if isinstance(self.guider, DynamicCFG) or isinstance(self.guider, TriangleCFG):

                # idx: 25, 24, 23, ..., 0
                if self.directly_use_idx_as_timestep:
                    sampling_step = idx  
                    # sampling_step: [25, 24, 23, ...., 0]

                elif self.fix_timestep_bug:
                    sampling_step = self.num_steps - idx
                    # sampling_step: [0, 1, 2, ......]

                else:
                    sampling_step = self.num_steps - timestep
                    # sampling_step: [-974, -934, -894, ..... ,0]
                
                # import pdb; pdb.set_trace()
                # # TODO: A Bug???
                # * In the beginning of sampling
                # timesteps: 999
                # self.num_steps: 25
                denoised = self.guider(
                    denoised, (1 - alpha_cumprod_sqrt**2) ** 0.5, step_index=sampling_step, scale=scale
                )  # TODO: Check the trend of the step_index
                # step_index: -974, -934, -894
            else:
                denoised = self.guider(denoised, (1 - alpha_cumprod_sqrt**2) ** 0.5, scale=scale)
        return denoised

    def sampler_step(
        self,
        alpha_cumprod_sqrt,
        next_alpha_cumprod_sqrt,
        denoiser,
        x,
        cond,
        uc=None,
        idx=None,
        timestep=None,
        scale=None,
        scale_emb=None,
    ):
        denoised = self.denoise(
            x, denoiser, alpha_cumprod_sqrt, cond, uc, timestep, idx, scale=scale, scale_emb=scale_emb
        ).to(torch.float32)

        a_t = ((1 - next_alpha_cumprod_sqrt**2) / (1 - alpha_cumprod_sqrt**2)) ** 0.5
        b_t = next_alpha_cumprod_sqrt - alpha_cumprod_sqrt * a_t

        x = append_dims(a_t, x.ndim) * x + append_dims(b_t, x.ndim) * denoised
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, scale=None, scale_emb=None):
        x, s_in, alpha_cumprod_sqrt, num_sigmas, cond, uc, timesteps = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            x = self.sampler_step(
                s_in * alpha_cumprod_sqrt[i],
                s_in * alpha_cumprod_sqrt[i + 1],
                denoiser,
                x,
                cond,
                uc,
                idx=self.num_steps - i,
                timestep=timesteps[-(i + 1)],
                scale=scale,
                scale_emb=scale_emb,
            )

        return x


class VPSDEDPMPP2MSampler(VideoDDIMSampler):
    def get_variables(self, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt=None):
        alpha_cumprod = alpha_cumprod_sqrt**2
        lamb = ((alpha_cumprod / (1 - alpha_cumprod)) ** 0.5).log()
        next_alpha_cumprod = next_alpha_cumprod_sqrt**2
        lamb_next = ((next_alpha_cumprod / (1 - next_alpha_cumprod)) ** 0.5).log()
        h = lamb_next - lamb

        if previous_alpha_cumprod_sqrt is not None:
            previous_alpha_cumprod = previous_alpha_cumprod_sqrt**2
            lamb_previous = ((previous_alpha_cumprod / (1 - previous_alpha_cumprod)) ** 0.5).log()
            h_last = lamb - lamb_previous
            r = h_last / h
            return h, r, lamb, lamb_next
        else:
            return h, None, lamb, lamb_next

    def get_mult(self, h, r, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt):
        mult1 = ((1 - next_alpha_cumprod_sqrt**2) / (1 - alpha_cumprod_sqrt**2)) ** 0.5 * (-h).exp()
        mult2 = (-2 * h).expm1() * next_alpha_cumprod_sqrt

        if previous_alpha_cumprod_sqrt is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def sampler_step(
        self,
        old_denoised,
        previous_alpha_cumprod_sqrt,
        alpha_cumprod_sqrt,
        next_alpha_cumprod_sqrt,
        denoiser,
        x,
        cond,
        uc=None,
        idx=None,
        timestep=None,
        scale=None,
        scale_emb=None,
        **additional_model_inputs
    ):
        denoised = self.denoise(
            x, denoiser, alpha_cumprod_sqrt, cond, uc, timestep, idx, scale=scale, scale_emb=scale_emb,
            **additional_model_inputs
        ).to(torch.float32)
        if idx == 1:
            return denoised, denoised

        h, r, lamb, lamb_next = self.get_variables(
            alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt
        )
        mult = [
            append_dims(mult, x.ndim)
            for mult in self.get_mult(h, r, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt)
        ]
        mult_noise = append_dims((1 - next_alpha_cumprod_sqrt**2) ** 0.5 * (1 - (-2 * h).exp()) ** 0.5, x.ndim)

        x_standard = mult[0] * x - mult[1] * denoised + mult_noise * torch.randn_like(x)
        if old_denoised is None or torch.sum(next_alpha_cumprod_sqrt) < 1e-14:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return x_standard, denoised
        else:
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d + mult_noise * torch.randn_like(x)

            x = x_advanced

        return x, denoised

    # TODO: Decreasing t_aug.
    def cond_aug_chunk_inference(self, x, prefix_frames, s_in, alpha_cumprod_sqrt, round_progress=None):

        # TODO: Wrong order.
        # alpha_cumprod_sqrt
        # tensor([0.0000, 0.0052, 0.0109, 0.0171, 0.0239, 0.0312, 0.0390, 0.0474, 0.0565,
        #        0.0661, 0.0764, 0.0873, 0.0988, 0.1110, 0.1239, 0.1374, 0.1516, 0.1664,
        #        0.1819, 0.1982, 0.2151, 0.2327, 0.2510, 0.2700, 0.2897, 0.3101, 0.3313,
        #        0.3531, 0.3757, 0.3990, 0.4231, 0.4478, 0.4733, 0.4996, 0.5265, 0.5541,
        #        0.5823, 0.6112, 0.6407, 0.6706, 0.7010, 0.7317, 0.7627, 0.7938, 0.8249,
        #        0.8558, 0.8864, 0.9164, 0.9456, 0.9739, 1.0000], device='cuda:0')

        if alpha_cumprod_sqrt[-1] > alpha_cumprod_sqrt[0]:
            # wrong order, fix it
            alpha_cumprod_sqrt = alpha_cumprod_sqrt.flip(0)
        
        if alpha_cumprod_sqrt[0] == 1.:
            alpha_cumprod_sqrt = alpha_cumprod_sqrt[1:]   # * Remove the first element - no corruption
        
        assert alpha_cumprod_sqrt[-1] < alpha_cumprod_sqrt[0]  
        # * Fixed value
        if isinstance(self.apply_cond_aug_chunk_inference, int):
            aug_t_chunk = self.apply_cond_aug_chunk_inference
            aug_t = aug_t_chunk + 50
            aug_t = int(aug_t / 1000 * len(alpha_cumprod_sqrt)) # scale to inference schedule

        elif self.apply_cond_aug_chunk_inference == 'min':
            aug_t_chunk, aug_t = 0, 0

        # elif self.apply_cond_aug_chunk_inference == 'dynamic':
        #     aug_t_chunk = int(round_progress * 6) * 100
        #     aug_t = aug_t_chunk + 50  # * maximum 650
        #     aug_t = int(aug_t / 1000 * len(alpha_cumprod_sqrt)) # scale to inference schedule
        
        # else:
        #     raise NotImplementedError

        rd = torch.randn_like(prefix_frames)
        if self.apply_cond_aug_chunk_inference == 'v1':
            log_cond_aug_dist = torch.distributions.Normal(-3.0, 0.5)  # * Following SVD
            log_cond_aug = log_cond_aug_dist.sample()
            cond_aug = torch.exp(log_cond_aug)
            # aug_input = aug_input + cond_aug * torch.randn_like(aug_input)
            noised_prefix_frames = prefix_frames + cond_aug * rd
            aug_t_chunk = 0 # just as a placeholder
        else:
            i = aug_t
            noised_prefix_frames = alpha_cumprod_sqrt[i] * prefix_frames + rd * append_dims(
                s_in * (1 - alpha_cumprod_sqrt[i] ** 2) ** 0.5, len(prefix_frames.shape)
            )

        x = torch.cat([noised_prefix_frames, x[:, self.fixed_frames :]], dim=1)
        return x, int(aug_t_chunk)

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, scale=None, scale_emb=None, fixed_frames=None,
                 **additional_model_inputs):

        fixed_frames = fixed_frames or self.fixed_frames

        x, s_in, alpha_cumprod_sqrt, num_sigmas, cond, uc, timesteps = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        if fixed_frames is not None:
            prefix_frames = x[:, :fixed_frames]  # ! Dimension index not right?
        old_denoised = None

        # TODO: Check here!
        for i in self.get_sigma_gen(num_sigmas):
            if fixed_frames is not None:
                if self.sdedit:
                    rd = torch.randn_like(prefix_frames)
                    noised_prefix_frames = alpha_cumprod_sqrt[i] * prefix_frames + rd * append_dims(
                        s_in * (1 - alpha_cumprod_sqrt[i] ** 2) ** 0.5, len(prefix_frames.shape)
                    )
                    x = torch.cat([noised_prefix_frames, x[:, fixed_frames :]], dim=1)
                elif self.apply_cond_aug_chunk_inference == 'v1':
                    x, aug_t_chunk = self.cond_aug_chunk_inference(x, prefix_frames, s_in, alpha_cumprod_sqrt)
                elif self.apply_cond_aug_chunk_inference != 'zero':  # * For str, should use !=, not 'is not'
                    x, aug_t_chunk = self.cond_aug_chunk_inference(x, prefix_frames, s_in, alpha_cumprod_sqrt)
                    additional_model_inputs['aug_t_chunk'] = aug_t_chunk
                else:
                    x = torch.cat([prefix_frames, x[:, fixed_frames :]], dim=1)
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * alpha_cumprod_sqrt[i - 1],
                s_in * alpha_cumprod_sqrt[i],
                s_in * alpha_cumprod_sqrt[i + 1],
                denoiser,
                x,
                cond,
                uc=uc,
                idx=self.num_steps - i,
                timestep=timesteps[-(i + 1)],
                scale=scale,
                scale_emb=scale_emb,
                **additional_model_inputs
            )

        if fixed_frames is not None:
            x = torch.cat([prefix_frames, x[:, fixed_frames :]], dim=1)

        return x


class VPODEDPMPP2MSampler(VideoDDIMSampler):
    def get_variables(self, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt=None):
        alpha_cumprod = alpha_cumprod_sqrt**2
        lamb = ((alpha_cumprod / (1 - alpha_cumprod)) ** 0.5).log()
        next_alpha_cumprod = next_alpha_cumprod_sqrt**2
        lamb_next = ((next_alpha_cumprod / (1 - next_alpha_cumprod)) ** 0.5).log()
        h = lamb_next - lamb

        if previous_alpha_cumprod_sqrt is not None:
            previous_alpha_cumprod = previous_alpha_cumprod_sqrt**2
            lamb_previous = ((previous_alpha_cumprod / (1 - previous_alpha_cumprod)) ** 0.5).log()
            h_last = lamb - lamb_previous
            r = h_last / h
            return h, r, lamb, lamb_next
        else:
            return h, None, lamb, lamb_next

    def get_mult(self, h, r, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt):
        mult1 = ((1 - next_alpha_cumprod_sqrt**2) / (1 - alpha_cumprod_sqrt**2)) ** 0.5
        mult2 = (-h).expm1() * next_alpha_cumprod_sqrt

        if previous_alpha_cumprod_sqrt is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def sampler_step(
        self,
        old_denoised,
        previous_alpha_cumprod_sqrt,
        alpha_cumprod_sqrt,
        next_alpha_cumprod_sqrt,
        denoiser,
        x,
        cond,
        uc=None,
        idx=None,
        timestep=None,
    ):
        denoised = self.denoise(x, denoiser, alpha_cumprod_sqrt, cond, uc, timestep, idx).to(torch.float32)
        if idx == 1:
            return denoised, denoised

        h, r, lamb, lamb_next = self.get_variables(
            alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt
        )
        mult = [
            append_dims(mult, x.ndim)
            for mult in self.get_mult(h, r, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt)
        ]

        x_standard = mult[0] * x - mult[1] * denoised
        if old_denoised is None or torch.sum(next_alpha_cumprod_sqrt) < 1e-14:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return x_standard, denoised
        else:
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d

            x = x_advanced

        return x, denoised

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, scale=None, **kwargs):
        x, s_in, alpha_cumprod_sqrt, num_sigmas, cond, uc, timesteps = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        old_denoised = None
        for i in self.get_sigma_gen(num_sigmas):
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * alpha_cumprod_sqrt[i - 1],
                s_in * alpha_cumprod_sqrt[i],
                s_in * alpha_cumprod_sqrt[i + 1],
                denoiser,
                x,
                cond,
                uc=uc,
                idx=self.num_steps - i,
                timestep=timesteps[-(i + 1)],
            )

        return x
