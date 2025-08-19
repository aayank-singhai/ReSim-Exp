#!/usr/bin/env bash

set -x


# cd /cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/IDM_nips

# ./dist_test /cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/IDM_nips/configs/cond3_non_expert/xx.py

timestamp=`date +"%y%m%d.%H%M%S"`

CODE_HOME=/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/IDM_nips

CONFIG=$1

# CHECKPOINT=$2
CHECKPOINT=${2:-"/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/IDM_nips/ckpts/IDM/waymo_bl_aug_weak_ep20.pth"}

# CHECKPOINT='/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/IDM_nips/ckpts/IDM/waymo_bl_aug_weak_ep20.pth'  # * ep20 works the best

# GPUS=$3
# GPUS=1
GPUS=8

WORK_DIR=$(echo ${CONFIG%.*} | sed -e "s/configs/work_dirs/g")
WORK_DIR=${WORK_DIR}_${timestamp}
echo "WORK_DIR: ${WORK_DIR}"

if [ ! -d ${WORK_DIR} ]; then
    mkdir -p ${WORK_DIR}
fi

PORT=${PORT:-23459}
export PYTHONPATH=$CODE_HOME:$PYTHONPATH

cd $CODE_HOME
python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/test.py $CONFIG $CHECKPOINT --launcher pytorch \
    --work-dir ${WORK_DIR}/test --eval traj \
    2>&1 | tee ${WORK_DIR}/test.${timestamp}.log
