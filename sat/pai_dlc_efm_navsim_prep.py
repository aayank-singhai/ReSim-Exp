import os
import time
import subprocess
import os

# EXP_NAME = "navsim_full_with_carla_data_with_token"
EXP_NAME = "GROUP_navsim_full_with_carla_data_with_no_token"

CONFIG = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/outputs_hdd/GROUP_navsim_full_with_carla_data_with_no_token/GROUP_navsim_full_main2_plan_30k_steps_split"
# CONFIG = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/configs/navsim_offline_rl/GROUP_ltf_init_on_trainset/ltf_init_on_trainset"
# CONFIG = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/configs/navsim_offline_rl/GROUP_ltf_init_on_testset/ltf_init_on_testset"
# CONFIG = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/configs/navsim_visual_planning/GROUP_navsim_full_with_carla_data_with_token/GROUP_navsim_full_main2_plan_30k_steps_split"

N_TOTAL = 60


TEMPLATE = f"""./dlc submit pytorchjob \
    --name={EXP_NAME}_<ID> \
    --command='bash <<-EOF
source /cpfs01/user/yangjiazhi/.bashrc
conda activate /cpfs01/user/yangjiazhi/my_envs/cogvid

cd /cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/scripts

python make_navsim_json_for_testing.py --folder_path {CONFIG}_<ID>

EOF' \
    --data_sources=d-dcbhw3bc7cixdnl4yu,d-3ysnral3r6911qmkqb,d-xu3mowvrld979aj9kl,d-3jltg36rwabtkilukp \
    --resource_id=quotanur5zmaur63 \
    --workspace_id=86989 \
    --priority=5 \
    --worker_image=pjlab-wulan-acr-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pjlab-eflops/yangjiazhi:yangjiazhi2 \
    --workers=1 \
    --worker_cpu=12 \
    --worker_memory=200Gi \
    --worker_shared_memory=200Gi \
    --worker_gpu=0
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