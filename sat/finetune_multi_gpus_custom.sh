set -x
T=`date +%m%d%H%M`

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/SwissArmyTransformer:${SCRIPT_DIR}:${PYTHONPATH}"
cd "${SCRIPT_DIR}"

MASTER_PORT=${MASTER_PORT:-28594}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RANK=${RANK:-0}

CFG=$1
GPUS=$2
NNODES=$3
SEED=${4:-42}  # * Need to change your seed if resume a model (trained with seed 42)

# * Get basename of CFG as exp name
EXP_NAME=$(basename $CFG)
EXP_NAME="${EXP_NAME%.yaml}"

WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/info_logs}"
mkdir -p "${WORK_DIR}"

python -m torch.distributed.launch \
    --nproc_per_node=${GPUS} \
    --nnodes=${NNODES} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --node_rank=${RANK} \
    train_video.py \
    --base=${CFG} \
    --seed=${SEED} \
    2>&1 | tee ${WORK_DIR}/${EXP_NAME}_$T.log
