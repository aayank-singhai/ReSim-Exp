#!/usr/bin/env bash
set -x

timestamp=`date +"%y%m%d.%H%M%S"`

WORK_DIR=$(dirname $(readlink -f "$0"))
CODE_HOME=/cpfs01/user/gaoshenyuan/code/IDM

CONFIG=configs/video_translator_flow_aeaug_noisy_0.5.py
CHECKPOINT=${WORK_DIR}/../epoch_2.pth

GPUS=$1
PORT=${PORT:-23459}
export PYTHONPATH=$CODE_HOME:$PYTHONPATH

cd $CODE_HOME
python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/test.py $CONFIG $CHECKPOINT --launcher pytorch \
    --work-dir ${WORK_DIR}/test --eval traj \
    2>&1 | tee ${WORK_DIR}/test.${timestamp}.log
