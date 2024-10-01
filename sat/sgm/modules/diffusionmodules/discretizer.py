from abc import abstractmethod
from functools import partial

import numpy as np
import torch

from ...modules.diffusionmodules.util import make_beta_schedule
from ...util import append_zero


def generate_roughly_equally_spaced_steps(num_substeps: int, max_step: int) -> np.ndarray:
    return np.linspace(max_step - 1, 0, num_substeps, endpoint=False).astype(int)[::-1]


class Discretization:
    def __call__(self, n, do_append_zero=True, device="cpu", flip=False, return_idx=False):
        if return_idx:
            sigmas, idx = self.get_sigmas(n, device=device, return_idx=return_idx)
        else:
            sigmas = self.get_sigmas(n, device=device, return_idx=return_idx)
        sigmas = append_zero(sigmas) if do_append_zero else sigmas
        if return_idx:
            return sigmas if not flip else torch.flip(sigmas, (0,)), idx   # * sigmas get flipped here
        else:
            return sigmas if not flip else torch.flip(sigmas, (0,))

    @abstractmethod
    def get_sigmas(self, n, device):
        pass


class EDMDiscretization(Discretization):
    def __init__(self, sigma_min=0.002, sigma_max=80.0, rho=7.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def get_sigmas(self, n, device="cpu"):
        ramp = torch.linspace(0, 1, n, device=device)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return sigmas


class LegacyDDPMDiscretization(Discretization):
    def __init__(
        self,
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps=1000,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        betas = make_beta_schedule("linear", num_timesteps, linear_start=linear_start, linear_end=linear_end)
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.to_torch = partial(torch.tensor, dtype=torch.float32)

    def get_sigmas(self, n, device="cpu"):
        if n < self.num_timesteps:
            timesteps = generate_roughly_equally_spaced_steps(n, self.num_timesteps)
            alphas_cumprod = self.alphas_cumprod[timesteps]
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        else:
            raise ValueError

        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        sigmas = to_torch((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        return torch.flip(sigmas, (0,))  # sigma_t: 14.4 -> 0.029


class ZeroSNRDDPMDiscretization(Discretization):
    def __init__(
        self,
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps=1000,
        shift_scale=1.0,  # noise schedule t_n -> t_m: logSNR(t_m) = logSNR(t_n) - log(shift_scale)
        keep_start=False,
        post_shift=False,
    ):
        super().__init__()
        if keep_start and not post_shift:
            linear_start = linear_start / (shift_scale + (1 - shift_scale) * linear_start)
        self.num_timesteps = num_timesteps
        betas = make_beta_schedule("linear", num_timesteps, linear_start=linear_start, linear_end=linear_end)
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.to_torch = partial(torch.tensor, dtype=torch.float32)

        # test_alphas_cumprod = self.to_torch(self.alphas_cumprod)
        # test_alphas_cumprod_sqrt = test_alphas_cumprod.sqrt()
        # test_weights = 1 / (1 - test_alphas_cumprod)
        # test_weights[:100]
        # tensor([1176.4403,  586.8622,  390.3312,  292.0723,  233.1140,  193.8129,
        #  165.7418,  144.6910,  128.3182,  115.2218,  104.5075,   95.5798,
        #   88.0264,   81.5532,   75.9439,   71.0360,   66.7065,   62.8586,
        #   59.4163,   56.3190,   53.5170,   50.9704,   48.6455,   46.5150,
        #   44.5553,   42.7468,   41.0727,   39.5185,   38.0719,   36.7221,
        #   35.4597,   34.2766,   33.1655,   32.1200,   31.1346,   30.2042,
        #   29.3244,   28.4913,   27.7010,   26.9506,   26.2370,   25.5576,
        #   24.9102,   24.2923,   23.7021,   23.1378,   22.5978,   22.0804,
        #   21.5844,   21.1084,   20.6513,   20.2120,   19.7894,   19.3826,
        #   18.9908,   18.6132,   18.2490,   17.8975,   17.5581,   17.2302,
        #   16.9132,   16.6066,   16.3099,   16.0225,   15.7442,   15.4744,
        #   15.2128,   14.9591,   14.7129,   14.4738,   14.2416,   14.0159,
        #   13.7966,   13.5833,   13.3758,   13.1739,   12.9774,   12.7860,
        #   12.5996,   12.4179,   12.2409,   12.0682,   11.8999,   11.7356,
        #   11.5753,   11.4189,   11.2661,   11.1169,   10.9712,   10.8288,
        #   10.6896,   10.5535,   10.4205,   10.2903,   10.1630,   10.0385,
        #    9.9166,    9.7972,    9.6804,    9.5660]) and more
        #  * Min-SNR will clamp the above weights to configured value (e.g. 5)

        # SNR shift
        if not post_shift:
            self.alphas_cumprod = self.alphas_cumprod / (shift_scale + (1 - shift_scale) * self.alphas_cumprod)

        self.post_shift = post_shift
        self.shift_scale = shift_scale

    def get_sigmas(self, n, device="cpu", return_idx=False):
        if n < self.num_timesteps:
            timesteps = generate_roughly_equally_spaced_steps(n, self.num_timesteps)
            alphas_cumprod = self.alphas_cumprod[timesteps]
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        else:
            raise ValueError

        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        alphas_cumprod = to_torch(alphas_cumprod)
        alphas_cumprod_sqrt = alphas_cumprod.sqrt()
        alphas_cumprod_sqrt_0 = alphas_cumprod_sqrt[0].clone()
        alphas_cumprod_sqrt_T = alphas_cumprod_sqrt[-1].clone()

        alphas_cumprod_sqrt -= alphas_cumprod_sqrt_T
        alphas_cumprod_sqrt *= alphas_cumprod_sqrt_0 / (alphas_cumprod_sqrt_0 - alphas_cumprod_sqrt_T)

        if self.post_shift:
            alphas_cumprod_sqrt = (
                alphas_cumprod_sqrt**2 / (self.shift_scale + (1 - self.shift_scale) * alphas_cumprod_sqrt**2)
            ) ** 0.5

        if return_idx:
            return torch.flip(alphas_cumprod_sqrt, (0,)), timesteps
        else:
            return torch.flip(alphas_cumprod_sqrt, (0,))
            # after flipping: sqrt(alpha_t): 0 -> 0.998 (not 1 in the end.)
            # ??? Why reverse this???? -- Answer: Flipped in DiscreteSampling --> Discretization
