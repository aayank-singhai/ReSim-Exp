python main.py test \
    --ckpt_path checkpoints/epoch=0.ckpt \
    --config config/reward.yaml \
    2>&1 | tee output_test.log