import os
import time
import subprocess

# ! NOTE: Run this script in dir: /cpfs01/user/yangjiazhi


TEMPLATE = f"""./dlc submit pytorchjob \
    --name=<CFG_NAME> \
    --command='bash <<-EOF
source /cpfs01/user/yangjiazhi/.bashrc
conda activate /cpfs01/user/yangjiazhi/my_envs/bevformer
cd /cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/IDM_nips
./dist_test.sh <CFG> /cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/IDM_nips/ckpts/IDM/waymo_bl_aug_strong_ep20.pth
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
INTERVAL = 10

GROUP_ROOT = "/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/IDM_nips/configs/cond3_resumed"

jsons_to_eval = []

for json_name in os.listdir(GROUP_ROOT):
    json_path = os.path.join(GROUP_ROOT, json_name)
    
    
    if json_path.endswith(".py"):
        jsons_to_eval.append(json_path)


print("Total jsons to evaluate:", len(jsons_to_eval))

for ind, json in enumerate(jsons_to_eval):

    print("Evaluating", json)

    command = TEMPLATE.replace("<CFG_NAME>", os.path.basename(json).split(".")[0]).replace("<CFG>", json)
    subprocess.run(command, shell=True)
    time.sleep(INTERVAL)

