#! /bin/bash

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

run_cmd="$environs python sample_video.py --base /cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/configs/cogvideox_2b_infer_custom.yaml"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"