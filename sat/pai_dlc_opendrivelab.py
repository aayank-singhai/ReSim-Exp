import os
import time
import subprocess

EXP_NAME = "GROUP_ltf_init_on_testset"
# CONFIG = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/configs/navsim_offline_rl/GROUP_ltf_init_on_trainset/ltf_init_on_trainset"
CONFIG = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/configs/navsim_offline_rl/GROUP_ltf_init_on_testset/ltf_init_on_testset"

N_TOTAL = 40


TEMPLATE = f"""./dlc submit pytorchjob \
    --name={EXP_NAME}_<ID> \
    --command='bash <<-EOF
source /cpfs01/user/yangjiazhi/.bashrc
conda activate /cpfs01/user/yangjiazhi/my_envs/cogvid2
cd /cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat
./inference_custom.sh {CONFIG}_<ID>.yaml
EOF' \
    --data_sources=d-ceg1efkflyg23vzhev,d-k8tu6brdfsyi2wj470,d-x135sr0ge79zzka647,d-fhse5rx2daxjz7k2t7 \
    --resource_id=quota6grhfqhj13o \
    --workspace_id=28250 \
    --priority=5 \
    --worker_image=pjlab-wulan-acr-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pjlab-eflops/yangjiazhi:yangjiazhi2 \
    --workers=1 \
    --worker_cpu=12 \
    --worker_memory=200Gi \
    --worker_shared_memory=200Gi \
    --worker_gpu=1
"""
INTERVAL = 5

for exp_id in range(N_TOTAL):

    command = TEMPLATE.replace("<ID>", str(exp_id))
    # exp_name = "scene_model_no_reg"
    # config_path = f"digitaltwin/config/multi_traversal_exps/{exp_name}.py"
    # output_dir = f"experiments/{task_name}/{exp_name}"
    # command = TEMPLATE.replace("<ID>", f"{task_name}-{exp_name}").replace("<CONFIG_PATH>", config_path).replace("<OUTPUT_DIR>", output_dir).replace("<TASK_NAME>", task_name)
    subprocess.run(command, shell=True)
    time.sleep(INTERVAL)