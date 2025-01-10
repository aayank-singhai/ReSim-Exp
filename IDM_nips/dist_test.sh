#!/usr/bin/env bash
set -x

timestamp=`date +"%y%m%d.%H%M%S"`

CODE_HOME=/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/IDM_nips

CONFIG=$1
CHECKPOINT=$2
GPUS=$3

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
