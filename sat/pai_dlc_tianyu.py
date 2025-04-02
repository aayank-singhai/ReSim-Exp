import os
import time
import subprocess
from pathlib import Path

from nuplan_scripts.utils.config import load_config

DIGITALTWIN_PATH = "/cpfs01/user/litianyu/projects/DigitalTwin"
USER_CPFS = "d-n1jeht2kz74zi0mtf1"
SHARED_CPFS = "d-ceg1efkflyg23vzhev"
SHARED_NAS = "d-x135sr0ge79zzka647"
WORKER_IMAGE = "pjlab-wulan-acr-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pjlab-eflops/litianyu:ws-250114"

TEMPLATE = f"""dlc submit pytorchjob \
    --name=EXP-<ID> \
    --command='cd {DIGITALTWIN_PATH}; bash multi_traversal_scripts/pai_run_base_benchmarking.sh <CONFIG_PATH> <OUTPUT_DIR> <TASK_NAME>' \
    --data_sources={USER_CPFS},{SHARED_CPFS},{SHARED_NAS} \
    --resource_id=quota6grhfqhj13o \
    --workspace_id=28250 \
    --priority=5 \
    --worker_image={WORKER_IMAGE} \
    --workers=1 \
    --worker_cpu=24 \
    --worker_memory=400Gi \
    --worker_shared_memory=400Gi \
    --worker_gpu=2
"""
INTERVAL = 60

# task_name = "base_benchmarking_v1"
# task_name = "mt_ablation_1_trv"
# task_name = "mt_ablation_2_trv"
task_name = "mt_ablation_3_trv"

exp_name = "scene_model_no_reg"
config_path = f"digitaltwin/config/multi_traversal_exps/{exp_name}.py"
output_dir = f"experiments/{task_name}/{exp_name}"
command = TEMPLATE.replace("<ID>", f"{task_name}-{exp_name}").replace("<CONFIG_PATH>", config_path).replace("<OUTPUT_DIR>", output_dir).replace("<TASK_NAME>", task_name)
subprocess.run(command, shell=True)
