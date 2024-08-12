set -x
T=`date +%m%d%H%M`

MASTER_PORT=${MASTER_PORT:-28594}
# MASTER_PORT=${MASTER_PORT:-28596}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RANK=${RANK:-0}

CFG=$1
GPUS=$2
NNODES=$3
SEED=${5:-42}  # * Need to change your seed if resume a model (trained with seed 42)

WORK_DIR="/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/runs/logs"

python -m torch.distributed.launch \
    --nproc_per_node=${GPUS} \
    --nnodes=${NNODES} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --node_rank=${RANK} \
    train_video.py \
    --base=${CFG} \
    --seed=${SEED} \
    2>&1 | tee ${WORK_DIR}/train_$T.log