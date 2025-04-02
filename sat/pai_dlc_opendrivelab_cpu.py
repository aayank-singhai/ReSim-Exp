import os
import time
import subprocess

# EXP_NAME = "GROUP_base_planner_cmd_on_trainset"
# EXP_NAME = "GROUP_ltf_on_trainset_round2"

# EXP_NAME = "GROUP_ltf_init_on_testset"
# EXP_NAME = "GROUP_add_carla_data_act"
# EXP_NAME = "GROUP_add_carla_data_free"
# EXP_NAME = "GROUP_add_carla_data_free2_early_carla"
EXP_NAME = "GROUP_add_carla_data_act"


# CONFIG = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/configs/navsim_offline_rl/GROUP_ltf_init_on_trainset/ltf_init_on_trainset"
# CONFIG = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/configs/navsim_offline_rl/GROUP_ltf_init_on_testset/ltf_init_on_testset"
# CONFIG = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/configs/navsim_offline_rl/GROUP_base_planner_cmd_on_trainset/base_planner_cmd_on_trainset"

# CONFIG = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/configs/navsim_offline_rl/GROUP_ltf_on_trainset_round2/ltf_on_trainset"
# CONFIG = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/configs/waymo_action_control/GROUP_add_carla_data_act/infer_waymo_with_carla_data_action_center_split"
# CONFIG = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/configs/waymo_action_control/GROUP_add_carla_data_free/infer_waymo_with_carla_data_free_center_split"
# CONFIG = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/configs/waymo_action_control/GROUP_add_carla_data_free2_early_carla/infer_waymo_with_carla_data_free2_center_split"
CONFIG = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/world_model_evaluate/configs/GROUP_add_carla_data_act/action_center_add_carla_data"

N_TOTAL = 20


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
    --worker_cpu=18 \
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