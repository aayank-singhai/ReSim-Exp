#! /bin/bash

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

CFG=$1
# SEED=${2:-42}

# run_cmd="$environs python sample_video.py --base=${CFG} --seed=${SEED}"

run_cmd="$environs python sample_video_vista_reward.py --base=${CFG}"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"