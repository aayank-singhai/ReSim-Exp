import os
import math
import argparse
from typing import List, Union
from tqdm import tqdm
from omegaconf import ListConfig
import imageio
from functools import partial

import torch
import numpy as np
from einops import rearrange
import torchvision.transforms as TT

from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint
from sat import mpu
from sat.data_utils import make_loaders
from sgm.util import get_obj_from_str, isheatmap, exists


from diffusion_video import SATVideoDiffusionEngine
from arguments import get_args
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode


def read_from_cli():
    cnt = 0
    try:
        while True:
            x = input("Please input English text (Ctrl-D quit): ")
            yield x.strip(), cnt
            cnt += 1
    except EOFError as e:
        pass


def read_from_file(p, rank=0, world_size=1):
    with open(p, "r") as fin:
        cnt = -1
        for l in fin:
            cnt += 1
            if cnt % world_size != rank:
                continue
            yield l.strip(), cnt


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N: Union[List, ListConfig], T=None, device="cuda"):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = np.repeat([value_dict["prompt"]], repeats=math.prod(N)).reshape(N).tolist()
            batch_uc["txt"] = np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N)).reshape(N).tolist()
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def save_video_as_grid_and_mp4(video_batch: torch.Tensor, save_path: str, fps: int = 5, args=None, key=None):
    os.makedirs(save_path, exist_ok=True)

    for i, vid in enumerate(video_batch):
        gif_frames = []
        for frame in vid:
            frame = rearrange(frame, "c h w -> h w c")
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            gif_frames.append(frame)
        now_save_path = os.path.join(save_path, f"{i:06d}.mp4")
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            for frame in gif_frames:
                writer.append_data(frame)

def save_text(text: str, save_path: str):
    txt_save_path = os.path.join(save_path, "text.txt")
    with open(txt_save_path, "w") as fout:
        fout.write(text)

def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    return arr

def get_all_file_from_dir(dir, ext=".pt"):
    out_files = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(ext):
                file_path = os.path.join(root, file)
                out_files.append(file_path)
    return out_files

@torch.no_grad()
def decode_latents(model, latents):
    if not isinstance(latents, list):
        latents = [latents]
    for latent_file in latents:
        if isinstance(latent_file, str):
            # latent_path
            latent = torch.load(latent_file).to(model.device)  # torch.Size([1, 16, 13, 90, 160])
            
            # * truncate
            # TODO: RM Hard Code
            # TRUNCATE = 5
            TRUNCATE = 7
            latent = latent[:, :, :TRUNCATE].contiguous()
            
            recon = model.first_stage_model.decode(latent).to(torch.float32)

            samples_x = recon.permute(0, 2, 1, 3, 4).contiguous()
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()

            save_path = latent_file.replace(".pt", ".mp4")
            if mpu.get_model_parallel_rank() == 0:
                save_video_as_grid_and_mp4(samples, save_path, fps=args.sampling_fps)

            del latent, recon, samples_x, samples

def sampling_main(args, model_cls):
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls

    load_checkpoint(model, args)
    model.eval()

    predictive_mode = False

    if args.input_type == "cli":
        data_iter = read_from_cli()
    elif args.input_type == "txt":
        rank, world_size = mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
        print("rank and world_size", rank, world_size)
        data_iter = read_from_file(args.input_file, rank=rank, world_size=world_size)
    elif args.input_type == "latent_dir":
        latent_files = get_all_file_from_dir(args.input_file)
        print("show latent_files\n", latent_files[:5])
        decode_latents(model, latent_files)
        return
    elif args.input_type == "dataset":
        data_class = get_obj_from_str(args.data_config["target"])
        create_dataset_function = partial(data_class.create_dataset_function, **args.data_config["params"])
        train_data, val_data, test_data = make_loaders(args, create_dataset_function)
        data_iter = val_data
        predictive_mode = True
    else:
        raise NotImplementedError
    
    # * Custom configs for sampling
    image_size = args.sampling_video_size  # * Remember set image size in your config
    n_prediction_round = args.n_prediction_round

    sample_func = model.sample
    T, H, W, C, F = args.sampling_num_frames, image_size[0], image_size[1], args.latent_channels, 8
    num_samples = [1]
    force_uc_zero_embeddings = ["txt"]
    device = model.device

    out_root = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/outputs"
    if isinstance(args.base, list):
        cfg_path = args.base[0]
    else:
        cfg_path = args.base
    cfg_name = os.path.basename(cfg_path).replace(".yaml", "")
    out_dir = os.path.join(out_root, cfg_name)
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for ind_batch, batch in enumerate(tqdm(data_iter)):
            if args.input_type != "dataset":
                text, _ = batch
            else:
                text = batch["txt"][0]

            if predictive_mode:
                x = batch["mp4"].to(device).to(model.dtype)
                x = x.permute(0, 2, 1, 3, 4).contiguous()
                z = model.encode_first_stage(x, batch)
                z = z.permute(0, 2, 1, 3, 4).contiguous()
                # torch.Size([1, 13, 16, 64, 112])
            else:
                z = None

            # reload model on GPU
            model.to(device)
            # print("rank:", rank, "start to process", text, cnt)
            # print("rank:", rank, "start to process", text, ind_batch)
            print("start to process", text, ind_batch)

            # TODO: broadcast image2video
            value_dict = {
                "prompt": text,
                "negative_prompt": "",
                "num_frames": torch.tensor(T).unsqueeze(0),  # TODO: Check what's the use of num_frames
            }

            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, num_samples
            )

            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    print(key, batch[key].shape)
                elif isinstance(batch[key], list):
                    print(key, [len(l) for l in batch[key]])
                else:
                    print(key, batch[key])
            
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=force_uc_zero_embeddings,
            )

            for k in c:
                if not k == "crossattn":
                    c[k], uc[k] = map(lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc))
            
            for index in range(args.batch_size):
                samples_tot = []
                for ind_round in range(n_prediction_round):
                    is_end = (ind_round == n_prediction_round - 1)

                    sampling_kwargs = dict()

                    if args.input_type == "dataset":
                        if ind_round == 0:
                            cond_inds = [0, 1, 2]
                        else:
                            cond_inds = [-3, -2, -1]   # * Auto-regressive

                        sampling_kwargs['prefix'] = z[:, cond_inds]

                    # reload model on GPUp
                    model.to(device)

                    samples_z = sample_func(
                        c,
                        uc=uc,
                        batch_size=1,
                        shape=(T, C, H // F, W // F),
                        **sampling_kwargs,
                    )
                    samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()  # !!! Check shape
                    # samples_z: torch.Size([1, 16, 13, 64, 112])

                    # Unload the model from GPU to save GPU memory
                    model.to("cpu")
                    torch.cuda.empty_cache()
                    first_stage_model = model.first_stage_model
                    first_stage_model = first_stage_model.to(device)

                    latent = 1.0 / model.scale_factor * samples_z

                    # Decode latent serial to save GPU memory
                    # TODO: Check here.
                    recons = []
                    loop_num = (T - 1) // 2
                    for i in range(loop_num):
                        if i == 0:
                            start_frame, end_frame = 0, 3
                        else:
                            start_frame, end_frame = i * 2 + 1, i * 2 + 3
                        if i == loop_num - 1:
                            clear_fake_cp_cache = True
                        else:
                            clear_fake_cp_cache = False
                        with torch.no_grad():
                            recon = first_stage_model.decode(
                                latent[:, :, start_frame:end_frame].contiguous(), clear_fake_cp_cache=clear_fake_cp_cache
                            )

                        recons.append(recon)

                    recon = torch.cat(recons, dim=2).to(torch.float32)
                    samples_x = recon.permute(0, 2, 1, 3, 4).contiguous()
                    samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu() # !!! Check shape
                    # samples: torch.Size([1, 49, 3, 512, 896])

                    # * Autoregressive rollout
                    if ind_round != 0:
                        samples = samples[:, 4 * (len(cond_inds) - 1) + 1 :]

                    samples_tot.append(samples)
                    save_path = os.path.join(
                        out_dir, str(ind_batch)
                    )

                    z = samples_z.permute(0, 2, 1, 3, 4).contiguous()

                    if is_end and mpu.get_model_parallel_rank() == 0:
                        samples_tot = torch.cat(samples_tot, dim=1)
                        save_video_as_grid_and_mp4(samples_tot, save_path, fps=args.sampling_fps)
                        save_text(text, save_path)


if __name__ == "__main__":
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    py_parser = argparse.ArgumentParser(add_help=False)
    known, args_list = py_parser.parse_known_args()

    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    del args.deepspeed_config
    args.model_config.first_stage_config.params.cp_size = 1
    args.model_config.network_config.params.transformer_args.model_parallel_size = 1
    args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False

    sampling_main(args, model_cls=SATVideoDiffusionEngine)
