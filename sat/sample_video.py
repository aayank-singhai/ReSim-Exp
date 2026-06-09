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
from datetime import datetime
import shutil


# * Optimize Sampling Quality for Short-term Sequence

# * Optimize Sampling Quality for long-term Rollouts




def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N: Union[List, ListConfig], T=None, device="cuda"):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = np.repeat([value_dict["prompt"]], repeats=math.prod(N)).reshape(N).tolist()
            batch_uc["txt"] = np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N)).reshape(N).tolist()
        elif key in value_dict.keys():
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def save_video_as_grid_and_mp4(video_batch: torch.Tensor, save_path: str, fps: int = 5, args=None, key=None, ind=None, foldername=None):
    os.makedirs(save_path, exist_ok=True)

    for i, vid in enumerate(video_batch):
        gif_frames = []
        for frame in vid:
            frame = rearrange(frame, "c h w -> h w c")

            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            gif_frames.append(frame)
        
        file_name = f"{i:06d}"

        if ind is not None:
            file_name = f"{ind}_{file_name}"

        if foldername is not None:
            file_name = f"folder-{foldername}_{file_name}"
        
        if key is not None:
            file_name  = f"{key}_{file_name}"

        file_name = file_name + ".mp4"

        now_save_path = os.path.join(save_path, file_name)
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            for frame in gif_frames:
                writer.append_data(frame)

def save_text(text: str, save_path: str):
    txt_save_path = os.path.join(save_path, "text.txt")
    with open(txt_save_path, "w") as fout:
        fout.write(text)

def save_traj(traj: torch.tensor, save_path: str):
    # Write the traj in the end of the txt file
    txt_save_path = os.path.join(save_path, "text.txt")
    traj = traj.cpu().numpy().tolist()
    traj = traj[0] # * bs = 1
    with open(txt_save_path, "a") as fout:
        fout.write("\n")
        for waypoint in traj:
            fout.write(str(waypoint) + "\n")

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
            TRUNCATE = 7
            latent = latent[:, :, :TRUNCATE].contiguous()
            
            recon = model.first_stage_model.decode(latent).to(torch.float32)

            samples_x = recon.permute(0, 2, 1, 3, 4).contiguous()
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()

            save_path = latent_file.replace(".pt", ".mp4")
            if mpu.get_model_parallel_rank() == 0:
                save_video_as_grid_and_mp4(samples, save_path, fps=args.sampling_fps)

            del latent, recon, samples_x, samples


# * Efficient Decoding
def get_reconstruction_from_latents(latent, T, first_stage_model):
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
            )   # torch.Size([1, 16, 3 (or 2), 64, 112])

        recons.append(recon)
    
    recons = torch.cat(recons, dim=2).to(torch.float32)  # [b, c, t, h, w]
    return recons


# For n_cond: 1
# - if n_round = 0, cond_inds = [0]
# - if n_round = 1, cond_inds = [-1]

# For n_cond: 2
# - if n_round = 0, cond_inds = [0, 1]
# - if n_round = 1, cond_inds = [-2, -1]

# For n_cond: 3
# - if n_round = 0, cond_inds = [0, 1, 2]
# - if n_round = 1, cond_inds = [-3, -2, -1]

def get_cond_inds(n_cond, n_round=0):
    if n_round == 0:
        return list(range(n_cond))
    else:
        return [-3, -2, -1]


def set_seed(seed: int):
    """Set the manual seed for reproducibility in PyTorch."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("!!!!!Setted seed to!!!!!!!", seed)

def sampling_main(args, model_cls):


    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls

    iteration = load_checkpoint(model, args)  # 20000
    model.eval()

    predictive_mode = False

    if args.input_type == "latent_dir":
        latent_files = get_all_file_from_dir(args.input_file)
        print("show latent_files\n", latent_files[:5])
        decode_latents(model, latent_files)
        return
    elif args.input_type == "dataset":
        data_class = get_obj_from_str(args.data_config["target"])
        create_dataset_function = partial(data_class.create_dataset_function, **args.data_config["params"])

        args.num_workers = 8  # * More workers
        train_data, val_data, test_data = make_loaders(args, create_dataset_function)
        data_iter = val_data  # * Checked, len(val_data) == 12146
        predictive_mode = True
    else:
        raise NotImplementedError
    
    # * Custom configs for sampling
    image_size = args.sampling_video_size  # * Remember set image size in your config
    n_prediction_round = args.n_prediction_round

    T, H, W, C, F = args.sampling_num_frames, image_size[0], image_size[1], args.latent_channels, 8
    num_samples = [1]
    
    force_uc_zero_embeddings = ["txt", "fut_traj", "with_human_drive_token"]


    APPLY_TRAJ = args.apply_traj  # * Default False
    SAVE_RECON = args.save_recon  # * Default True, Turn to false to speed up sampling
    SAVE_GT = args.save_gt
    
    CONCAT_RECON_FOR_DEMO = args.concat_recon_for_demo  # * Default False
    CONCAT_GT_FOR_DEMO = args.concat_gt_for_demo  # * Default False
    N_COND_FRAMES = args.n_cond_frames  # * Default 3

    device = model.device

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_root = os.path.join(repo_root, "outputs")

    if isinstance(args.base, list):
        cfg_path = args.base[0]
    else:
        cfg_path = args.base

    if "GROUP" not in cfg_path:
        cfg_name = os.path.basename(cfg_path).replace(".yaml", "")
    else:
        cfg_name = os.path.join(*cfg_path.split("/")[-2:]).replace(".yaml", "")

    out_dir = os.path.join(out_root, cfg_name)

    if "GROUP" not in cfg_path:
        out_dir = out_dir + '-' +datetime.now().strftime("%m-%d-%H-%M")
    os.makedirs(out_dir, exist_ok=True)

    # write iteration to file
    with open(os.path.join(out_dir, "iteration.txt"), "w") as fout:
        fout.write(str(iteration))

    shutil.copyfile(cfg_path, os.path.join(out_dir, os.path.basename(cfg_path)))

    with torch.no_grad():
        for ind_batch, batch in enumerate(tqdm(data_iter)):
            
            print(f"Processing {ind_batch} / {len(data_iter)} batch")


            if args.input_type != "dataset":
                text, _ = batch
            else:
                text = batch["txt"][0]

            if predictive_mode:
                x = batch["mp4"].to(device).to(model.dtype)
                input_video = x.clone()  # [b, t, c, h, w]

                x = x.permute(0, 2, 1, 3, 4).contiguous()

                z = model.encode_first_stage(x, batch)
                z_origin = z.clone()  # * For logging

                z = z.permute(0, 2, 1, 3, 4).contiguous()
            else:
                z = None
                z_origin = None

            # reload model on GPU
            model.to(device)
            print("start to process", text, ind_batch)

            value_dict = {
                "mp4": batch["mp4"].to(device),
                "prompt": text,
                "negative_prompt": "",
                "num_frames": torch.tensor(T).unsqueeze(0),
            }

            if "fut_traj" in batch:
                value_dict["fut_traj"] = batch["fut_traj"].to(device)

            if "with_human_drive_token" in batch:
                value_dict["with_human_drive_token"] = batch["with_human_drive_token"].to(device)

            if "lidar_pc_token" in batch:
                lidar_pc_tokens = batch["lidar_pc_token"]
            else:
                print(f"No lidar_pc_token in batch, generating a new token: {ind_batch}")
                lidar_pc_tokens = [str(ind_batch)]
            
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
                force_c_zero_embeddings=[] if APPLY_TRAJ else ["fut_traj"],
                force_uc_zero_embeddings=force_uc_zero_embeddings,
            )

            for k in c:
                if not k == "crossattn":
                    c[k], uc[k] = map(lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc))
            
            for index in range(args.batch_size):
                lidar_pc_token = lidar_pc_tokens[index]
                samples_tot = []
                for ind_round in range(n_prediction_round):

                    sampling_kwargs = dict()
                    sampling_kwargs['round_progress'] = ind_round / n_prediction_round

                    if args.input_type == "dataset":
                        cond_inds = get_cond_inds(N_COND_FRAMES, ind_round)
                        sampling_kwargs['prefix'] = z[:, cond_inds]
                        
                        sampling_kwargs['cond_inds'] = cond_inds if ind_round == 0 else [0, 1, 2]
                        
                    # reload model on GPUp
                    model.to(device)

                    samples_z = model.sample(
                        c,
                        uc=uc,
                        batch_size=1,
                        shape=(T, C, H // F, W // F),
                        **sampling_kwargs,
                    )  # * Go to SATVideoDiffusionEngine sample func
                    samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()
                    # samples_z: torch.Size([1, 16, 13, 64, 112])

                    # Unload the model from GPU to save GPU memory
                    model.to("cpu")
                    torch.cuda.empty_cache()
                    first_stage_model = model.first_stage_model
                    first_stage_model = first_stage_model.to(device)

                    latent = 1.0 / model.scale_factor * samples_z

                    # Decode latent serial to save GPU memory
                    # Proved: Do we need this?? If so, can we apply this in training to reduce memory?
                    # Replacing the slided decoding to full decoding is not that helpful.

                    samples = get_reconstruction_from_latents(latent, T, first_stage_model)  # [b, c, t, h, w]
                    samples = samples.permute(0, 2, 1, 3, 4).contiguous()
                    samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0).cpu()
                    # samples: torch.Size([1, 49, 3, 512, 896])

                    # * Autoregressive rollout
                    if ind_round != 0:
                        samples = samples[:, 4 * (len(cond_inds) - 1) + 1 :]

                    samples_tot.append(samples)
                    save_path = os.path.join(
                        out_dir, str(ind_batch)
                    )

                    z = samples_z.permute(0, 2, 1, 3, 4).contiguous()

                if mpu.get_model_parallel_rank() == 0:

                    assert not SAVE_RECON and not CONCAT_RECON_FOR_DEMO, "save memory, do not save recon or concat_recon_for_demo"
                    if SAVE_RECON:
                        # gt_rec = model.decode_first_stage(z_origin).to(torch.float32)  # * Memory consuming
                        gt_rec = get_reconstruction_from_latents(z_origin, T, first_stage_model)  # * Memory efficient mode
                        gt_rec = gt_rec.permute(0, 2, 1, 3, 4).contiguous()
                        gt_rec = torch.clamp((gt_rec + 1.0) / 2.0, min=0.0, max=1.0).cpu()
                        save_video_as_grid_and_mp4(gt_rec, save_path, fps=args.sampling_fps, key="Rec", ind=lidar_pc_token, foldername=str(ind_batch))

                    if SAVE_GT:
                        input_video = input_video = (input_video + 1.0) / 2.0
                        input_video = input_video.cpu()
                        save_video_as_grid_and_mp4(input_video, save_path, fps=args.sampling_fps, key="GT", ind=lidar_pc_token, foldername=str(ind_batch))

                    # Sample
                    samples_tot = torch.cat(samples_tot, dim=1)
                    save_video_as_grid_and_mp4(samples_tot, save_path, fps=args.sampling_fps, key="Sample", ind=lidar_pc_token, foldername=str(ind_batch))
                    save_text(text, save_path)
                    if APPLY_TRAJ and "fut_traj" in batch:
                        save_traj(value_dict["fut_traj"], save_path)
                    
                    # Concatenate gt for demo
                    if CONCAT_RECON_FOR_DEMO:
                        assert SAVE_RECON
                        print("sample_tot shape", samples_tot.shape)   # [1, 49, 3, 512, 896]
                        print("gt_rec shape", gt_rec.shape)            # [1, 49, 3, 512, 896]
                        concated = torch.cat([gt_rec, samples_tot], dim=-1)  # * Left: GT | Right: Sample
                        save_video_as_grid_and_mp4(concated, save_path, fps=args.sampling_fps, key="Concated", ind=lidar_pc_token, foldername=str(ind_batch))

                    if CONCAT_GT_FOR_DEMO:
                        assert SAVE_GT
                        concated = torch.cat([input_video, samples_tot], dim=-1)  # * Left: GT | Right: Sample
                        save_video_as_grid_and_mp4(concated, save_path, fps=args.sampling_fps, key="ConcatedGT", ind=lidar_pc_token, foldername=str(ind_batch))

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
