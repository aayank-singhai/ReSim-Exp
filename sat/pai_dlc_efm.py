import os
import time
import subprocess

# EXP_NAME = "GROUP_abl_no_carla_data_act"

# EXP_NAME = "GROUP_abl_no_carla_data_free"
EXP_NAME = "GROUP_abl_with_carla_data_act"
# EXP_NAME = "GROUP_abl_with_carla_data_free"

# CONFIG = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/configs/navsim_offline_rl/GROUP_ltf_init_on_trainset/ltf_init_on_trainset"
# CONFIG = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/configs/navsim_offline_rl/GROUP_ltf_init_on_testset/ltf_init_on_testset"
# CONFIG = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/configs/navsim_visual_planning/GROUP_navsim_full_with_carla_data_with_token/GROUP_navsim_full_main2_plan_30k_steps_split"
# CONFIG = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/configs/waymo_action_control/GROUP_abl_no_carla_data_act/abl_no_carla_data_action_center_split"

# CONFIG = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/configs/waymo_action_control/GROUP_abl_no_carla_data_free/abl_no_carla_data_free_center_split"
CONFIG = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/configs/waymo_action_control/GROUP_abl_with_carla_data_act/infer_waymo_with_carla_data_action_center_split"
# CONFIG = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/configs/waymo_action_control/GROUP_abl_with_carla_data_free/infer_waymo_abl_with_carla_data_free_center_split"

N_TOTAL = 20


TEMPLATE = f"""./dlc submit pytorchjob \
    --name={EXP_NAME}_<ID> \
    --command='bash <<-EOF
source /cpfs01/user/yangjiazhi/.bashrc
conda activate /cpfs01/user/yangjiazhi/my_envs/cogvid2
cd /cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat
./inference_custom.sh {CONFIG}_<ID>.yaml
EOF' \
    --data_sources=d-dcbhw3bc7cixdnl4yu,d-3ysnral3r6911qmkqb,d-xu3mowvrld979aj9kl,d-3jltg36rwabtkilukp \
    --resource_id=quotanur5zmaur63 \
    --workspace_id=86989 \
    --priority=9 \
    --worker_image=pjlab-wulan-acr-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pjlab-eflops/yangjiazhi:yangjiazhi2 \
    --workers=1 \
    --worker_cpu=12 \
    --worker_memory=200Gi \
    --worker_shared_memory=200Gi \
    --worker_gpu=1
"""
INTERVAL = 10

for exp_id in range(N_TOTAL):

    command = TEMPLATE.replace("<ID>", str(exp_id))
    # exp_name = "scene_model_no_reg"
    # config_path = f"digitaltwin/config/multi_traversal_exps/{exp_name}.py"
    # output_dir = f"experiments/{task_name}/{exp_name}"
    # command = TEMPLATE.replace("<ID>", f"{task_name}-{exp_name}").replace("<CONFIG_PATH>", config_path).replace("<OUTPUT_DIR>", output_dir).replace("<TASK_NAME>", task_name)
    subprocess.run(command, shell=True)
    time.sleep(INTERVAL)