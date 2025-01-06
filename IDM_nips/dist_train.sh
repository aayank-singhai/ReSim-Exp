#!/usr/bin/env bash
set -x

timestamp=`date +"%y%m%d.%H%M%S"`

CODE_HOME=/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/IDM_nips

CONFIG=$1
GPUS=$2

WORK_DIR=$(echo ${CONFIG%.*} | sed -e "s/configs/work_dirs/g")/
echo "WORK_DIR: ${WORK_DIR}"

if [ ! -d ${WORK_DIR} ]; then
    mkdir -p ${WORK_DIR}
fi

PORT=${PORT:-28510}
export PYTHONPATH=$CODE_HOME:$PYTHONPATH

cd $CODE_HOME
python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CONFIG --launcher pytorch --work-dir ${WORK_DIR} \
    2>&1 | tee ${WORK_DIR}/train.${timestamp}.log
