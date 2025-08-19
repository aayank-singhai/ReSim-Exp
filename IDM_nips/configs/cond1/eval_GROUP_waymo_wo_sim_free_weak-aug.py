_base_ = []
custom_imports = dict(imports=['plugins'])

dataset_type = 'WaymoTranslationDataset'
queue_length = 5
condition_frames = 1


gen_image_root = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/outputs_hdd/GROUP_Action_Control_wo_sim_1cond_fixed/GROUP_waymo_wo_sim_free_1cond'
infer_ann_file = '/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/waymo/v2/waymo_val_traj_cmd_v2_mini_sub_uniform_540.json'


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

data_root = '/cpfs01/shared/opendrivelab/opendrivelab_hdd/GenAD_Proj/ad_datasets/Waymo/kitti_format/training/images_0'
model = dict(
    type='VideoTranslatorFlow',
    queue_length=queue_length,
    condition_frames=condition_frames,
    feedforward_channels=256
)

train_pipeline = [


    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=False),
    dict(type='CenterCropResizeMultiViewImage', scale=(512, 896)),  # * Align with diffusion model sampling
    dict(type='CenterCropResizeMultiViewImage', scale=(384, 640)),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='CustomDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_traj'], meta_keys=[])
]

test_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=False),
    dict(type='CenterCropResizeMultiViewImage', scale=(384, 640)),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='CustomDefaultFormatBundle'),
    dict(type='Collect', keys=['img'], meta_keys=[])
]



data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/cpfs01/user/yangjiazhi/workspace/DVGen/CogVideo/custom_data/waymo/train/waymo_train_traj_cmd_v2.json',
        pipeline=train_pipeline,
        load_interval=5,  # 10 hz -> 2hz
        queue_length=queue_length,
        condition_frames=condition_frames,
        test_mode=False),
    val=dict(
        type='WaymoTranslationDatasetEval',
        data_root=data_root,
        ann_file=infer_ann_file,
        
        gen_image_root=gen_image_root,
        
        pipeline=test_pipeline,
        load_interval=1,
        queue_length=queue_length,
        condition_frames=condition_frames,
        test_mode=True,
        sample_key="GT"
        ),  # * Val on GT
    test=dict(
        type='WaymoTranslationDatasetEval',
        data_root=data_root,
        ann_file=infer_ann_file,

        gen_image_root=gen_image_root,

        pipeline=test_pipeline,
        load_interval=1,
        queue_length=queue_length,
        condition_frames=condition_frames,
        test_mode=True,
        sample_key="Sample"
        ),
    test_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=0, shuffle=False)
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 10
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook')
    ])

checkpoint_config = dict(interval=1, max_keep_ckpts=1)

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = 'ckpts/XVO_translator.pth'
resume_from = None
workflow = [('train', 1)]
# find_unused_parameters = True
