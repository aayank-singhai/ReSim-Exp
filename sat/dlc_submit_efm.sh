dlc submit pytorchjob \
    --name=base_planner_cmd_on_trainset_0_clone \
    --command='bash <<-EOF
source /cpfs01/user/yangjiazhi/.bashrc
conda activate /cpfs01/user/yangjiazhi/my_envs/cogvid2
cd /cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat
./inference_custom.sh /cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/sat/configs/navsim_offline_rl/GROUP_base_planner_cmd_on_trainset/base_planner_cmd_on_trainset_0.yaml
EOF' \
    --data_sources=d-dcbhw3bc7cixdnl4yu,d-3ysnral3r6911qmkqb,d-xu3mowvrld979aj9kl,d-3jltg36rwabtkilukp \
    --resource_id=quotanur5zmaur63 \
    --tags="CloneFromJobID=dlcu7odtzd6ykta9" \
    --workspace_id=86989 \
    --vpc_id=vpc-0jluo8j06nwlb1lohoi4s \
    --switch_id=vsw-0jlgv2dllms680bsohusz \
    --security_group_id=sg-0jl8b4mijll7mgqmyk71 \
    --priority=5 \
    --extended_cidrs="10.3.255.208/28,10.2.0.0/24,10.3.255.240/28,10.3.255.224/28,10.0.0.0/16" \
    --job_max_running_time_minutes=43200 \
    --workers=1 \
    --worker_image=pjlab-wulan-acr-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pjlab-eflops/yangjiazhi:yangjiazhi2 \
    --worker_cpu=12 \
    --worker_memory=200Gi \
    --worker_shared_memory=200Gi \
    --worker_gpu=1 