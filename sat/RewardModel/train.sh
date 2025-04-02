torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    main.py fit \
    --config config/reward.yaml \
    2>&1 | tee output_train.log