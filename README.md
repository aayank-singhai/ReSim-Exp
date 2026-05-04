# ReSim

**Reliable World Simulation for Autonomous Driving**

ReSim is a driving world model for simulating future ego-view driving videos under
different action conditions, including expert, free-driving, commanded, and
hazardous non-expert behaviors.

> Technical report: https://arxiv.org/abs/2506.09981
>
> Video demos: https://opendrivelab.com/ReSim
>
> Primary contact: Jiazhi Yang, jzyang@link.cuhk.edu.hk

## Overview

ReSim fine-tunes a CogVideoX/SAT-style video diffusion model for autonomous
driving simulation. The central idea is to co-train on heterogeneous driving
sources: large-scale web driving videos, real driving datasets with action or
trajectory labels, and simulated CARLA data that contains non-expert behavior.

This repository currently contains research code rather than a polished library.
Most workflows are config-driven and several checked-in configs still contain
internal absolute paths. Before running an experiment, copy a config and replace
all dataset, checkpoint, log, and output paths with paths on your machine.

## Repository Layout

```text
.
|-- sat/                    # ReSim world-model training and inference code
|   |-- configs/            # Example training and inference configs
|   |-- sgm/                # Diffusion, conditioning, sampling, and VAE modules
|   |-- train_video.py      # Main world-model training entrypoint
|   `-- sample_video.py     # Main world-model inference entrypoint
|-- inference/              # Upstream CogVideoX text-to-video demos
|-- tools/                  # Weight conversion and captioning utilities
|-- SwissArmyTransformer/   # Vendored SAT dependency used by training/inference
|-- requirements.txt        # Top-level demo/runtime dependencies
`-- sat/requirements.txt    # SAT world-model dependencies
```

## Installation

The main code path was developed with Python 3.10, PyTorch 2.4, CUDA 12.4, and
SAT/DeepSpeed-style distributed training.

```bash
conda create -n resim python=3.10 -y
conda activate resim

pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -r sat/requirements.txt

cd SwissArmyTransformer
pip install -e .
cd ..
```

Notes:

- `sat/requirements.txt` includes the heavier training stack, including
  DeepSpeed, OmegaConf, PyTorch Lightning, decord, wandb, and SAT-related
  packages.
- Some utilities use optional services such as OpenAI-compatible APIs or wandb;
  those are not required for basic world-model training or inference.

## Checkpoints

ReSim builds on CogVideoX-2B/SAT components. A typical checkpoint directory
contains:

```text
CogVideoX-2b-sat/
|-- transformer/
|   |-- latest
|   `-- <iteration>/mp_rank_00_model_states.pt
|-- vae/3d-vae.pt
`-- t5-v1_1-xxl/
    |-- config.json
    |-- model-00001-of-00002.safetensors
    |-- model-00002-of-00002.safetensors
    |-- model.safetensors.index.json
    |-- spiece.model
    `-- tokenizer_config.json
```

Update the following fields in your copied config before running:

- `args.load`: transformer checkpoint or ReSim fine-tuned checkpoint directory.
- `model.conditioner_config...FrozenT5Embedder.params.model_dir`: T5 directory.
- `model.first_stage_config.params.ckpt_path`: VAE checkpoint path.
- `args.train_data` and `args.valid_data`: dataset annotation paths.

## Data Format

The world-model loaders are JSON-driven. Real driving and simulator datasets use
the shared format consumed by `sat/data_share.py`:

```json
{
  "meta": {
    "data_root": "/path/to/image/root"
  },
  "clips": [
    {
      "img_seq": ["scene/frame_000.jpg", "scene/frame_001.jpg"],
      "cmd": "Moving_Forward",
      "traj_fut": [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0]],
      "lidar_pc_token": "sample-token"
    }
  ]
}
```

Important fields:

- `img_seq` is a list of frame paths relative to `meta.data_root`. The loader
  also supports `img_seq_his` plus `img_seq_fut`.
- `cmd` can be a string such as `Moving_Forward`, `Turning_Left`,
  `Turning_Right`, or an integer mapped by `sat/data_utils.py`.
- `traj_fut` stores future trajectory points, expected as `[x, y, heading]`.
  The default configs use 8 future points.
- `lidar_pc_token` or `token` is used to name generated outputs.

For web-driving data, `sat/data_youtube.py` expects clips with
`folder_name`, `first_frame`, `end_frame`, and `flow_direction`.

## World-Model Training

Use `sat/train_video.py` through the helper shell script:

```bash
cd sat

# CFG, GPUS, NNODES, optional SEED
bash finetune_multi_gpus_custom.sh configs/main5_joint_stage2_high_small-lr_full.yaml 8 1 42
```

For single-GPU debugging:

```bash
cd sat
bash finetune_single_gpu_custom.sh configs/main5_joint_stage2_high_small-lr_full.yaml
```

Before launching, edit the config paths and check:

- `args.mode: finetune`
- `data.target`, for example `data_multi.MultiSourceDataset` or
  `data_waymo.WaymoDataset`
- `data.params.video_size`, `fps`, `max_num_frames`, and crop mode
- DeepSpeed batch size, gradient accumulation, precision, and save interval
- `train_data_weights` when mixing heterogeneous data sources

Training writes checkpoints under the config's `args.save` directory and stores
a copy of the merged training config with the run.

## World-Model Inference

Use `sat/sample_video.py` through `sat/inference_custom.sh`:

```bash
cd sat
bash inference_custom.sh configs/infer_nus_new.yaml
```

The provided inference config uses `input_type: dataset`; it loads validation
clips, conditions on the first frames, optionally applies `fut_traj`, and writes
MP4 samples. In the current code, `sample_video.py` also contains an internal
default output root, so update it or patch the script if you need a custom
output location for a public release.

Common inference options are config-driven:

- `args.sampling_video_size`: output frame size, for example `[512, 896]`.
- `args.sampling_num_frames`: latent-frame count, commonly `13`, `11`, or `9`.
- `args.n_prediction_round`: autoregressive rollout rounds.
- `args.apply_traj`: whether to condition on `fut_traj`.
- `args.save_gt` and `args.concat_gt_for_demo`: whether to save ground truth and
  side-by-side demo videos.

## Auxiliary Components

- `tools/convert_weight_sat2hf.py`: converts SAT-format transformer/VAE
  checkpoints into Hugging Face Diffusers-compatible CogVideoX weights.
- `inference/`: upstream CogVideoX prompt-to-video demos; these are useful for
  sanity-checking dependencies but are separate from the ReSim driving
  world-model pipeline.

## Public Release Checklist

Before publishing this repository, review the following:

- Replace or remove internal absolute paths under `/cpfs01`, `/inspire`, and
  user home directories from configs and scripts.
- Remove generated artifacts, caches, logs, local outputs, and private
  checkpoints unless they are intended for release.
- Publish model checkpoints and data annotations separately, or clearly mark
  them as unavailable.
- Verify that dataset licenses permit redistribution or document how users
  should obtain the datasets themselves.
- Re-run at least one minimal inference job from a clean checkout.

## Citation

If this project is useful for your research, please cite:

```bibtex
@inproceedings{yang2025resim,
  title={ReSim: Reliable World Simulation for Autonomous Driving},
  author={Jiazhi Yang and Kashyap Chitta and Shenyuan Gao and Long Chen and Yuqian Shao and Xiaosong Jia and Hongyang Li and Andreas Geiger and Xiangyu Yue and Li Chen},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

## License

The repository includes an Apache-2.0 `LICENSE` file inherited from the
CogVideo/CogVideoX codebase. Model weights are governed by `MODEL_LICENSE`.
Check the licenses of CogVideoX, SAT, CARLA, nuScenes, Waymo, nuPlan, OpenDV,
and any redistributed annotations before public release or commercial use.
