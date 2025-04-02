import os
import time
import subprocess
import os

# EXP_NAME = "navsim_full_with_carla_data_with_token"
# EXP_NAME  = "GROUP_navsim_full_with_carla_data_with_no_token"
EXP_NAME="GROUP_navsim_full_main2_plan_30k_steps"

# CONFIG = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/configs/navsim_offline_rl/GROUP_ltf_init_on_trainset/ltf_init_on_trainset"
# CONFIG = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/configs/navsim_offline_rl/GROUP_ltf_init_on_testset/ltf_init_on_testset"
# CONFIG = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/configs/navsim_visual_planning/GROUP_navsim_full_with_carla_data_with_token/GROUP_navsim_full_main2_plan_30k_steps_split"

# N_TOTAL = 60

TEMPLATE = f"""./dlc submit pytorchjob \
    --name={EXP_NAME}_<ID> \
    --command='bash <<-EOF
source /cpfs01/user/yangjiazhi/.bashrc
conda activate /cpfs01/user/yangjiazhi/my_envs/navsim

cd /cpfs01/user/yangjiazhi/workspace/DVGen/NavSim_WorkSpace/navsim/scripts/evaluation

./eval_new_idm_with_gen.sh <JSON>

EOF' \
    --data_sources=d-k8tu6brdfsyi2wj470,d-ceg1efkflyg23vzhev,d-x135sr0ge79zzka647,d-fhse5rx2daxjz7k2t7 \
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
INTERVAL = 10

# GROUP_ROOT = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/outputs_hdd/GROUP_navsim_full_with_carla_data_with_token"
# GROUP_ROOT =   "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/outputs_hdd/GROUP_navsim_full_with_carla_data_with_no_token"
GROUP_ROOT = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/outputs_hdd/navsim_visual_planning/GROUP_navsim_full_main2_plan_30k_steps"

jsons_to_eval = []

for group_folder in os.listdir(GROUP_ROOT):
    group_folder_path = os.path.join(GROUP_ROOT, group_folder)
    
    # * Skip files if not a folder
    if not os.path.isdir(group_folder_path):
        continue

    for subfile in os.listdir(group_folder_path):
        if subfile.endswith(".json") and subfile.count("split") == 1:
            jsons_to_eval.append(os.path.join(group_folder_path, subfile))


print("Total jsons to evaluate:", len(jsons_to_eval))

for ind, json in enumerate(jsons_to_eval):

    print("Evaluating", json)

    command = TEMPLATE.replace("<JSON>", json).replace("<ID>", str(ind))
    subprocess.run(command, shell=True)
    time.sleep(INTERVAL)
