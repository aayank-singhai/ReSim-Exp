#! /bin/bash
set -euo pipefail

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/SwissArmyTransformer:${SCRIPT_DIR}:${PYTHONPATH:-}"
cd "${SCRIPT_DIR}"

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

CFG=$1


run_cmd="$environs python sample_video.py --base=${CFG}"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"