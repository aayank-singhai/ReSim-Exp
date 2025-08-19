set -x
T=`date +%m%d%H%M`

CFG=$1

EXP_NAME=$(basename $CFG)
EXP_NAME="${EXP_NAME%.yaml}"
WORK_DIR="/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/world_model_evaluate/logs"

python eval_v3.py --yaml ${CFG} \
       2>&1 | tee ${WORK_DIR}/${EXP_NAME}_$T.log