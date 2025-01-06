custom_imports = dict(imports=['plugins'])
dataset_type = 'NuScenesTranslationDataset'
queue_length = 5
condition_frames = 1
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_root = '/cpfs01/shared/opendrivelab/nuscenes/'
model = dict(
    type='VideoTranslatorFlow',
    queue_length=5,
    condition_frames=1,
    feedforward_channels=256)
train_pipeline = [
    dict(
        type='UseAutoEncoderData',
        data_root=
        '/cpfs01/shared/opendrivelab/opendrivelab_hdd/gaoshenyuan/IDM_samples_new',
        p_noisy=0.5),
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=False),
    dict(type='CenterCropResizeMultiViewImage', scale=(384, 640)),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='CustomDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_traj'], meta_keys=[])
]
test_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=False),
    dict(type='CenterCropResizeMultiViewImage', scale=(384, 640)),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='CustomDefaultFormatBundle'),
    dict(type='Collect', keys=['img'], meta_keys=[])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type='NuScenesTranslationDataset',
        data_root='/cpfs01/shared/opendrivelab/nuscenes/',
        ann_file=
        '/cpfs01/shared/opendrivelab/nuscenes/nuscenes_infos_temporal_train.pkl',
        pipeline=[
            dict(
                type='UseAutoEncoderData',
                data_root=
                '/cpfs01/shared/opendrivelab/opendrivelab_hdd/gaoshenyuan/IDM_samples_new',
                p_noisy=0.5),
            dict(type='CustomLoadMultiViewImageFromFiles', to_float32=False),
            dict(type='CenterCropResizeMultiViewImage', scale=(384, 640)),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='CustomDefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_traj'], meta_keys=[])
        ],
        load_interval=1,
        queue_length=5,
        condition_frames=1,
        test_mode=False),
    val=dict(
        type='NuScenesTranslationDataset',
        data_root='/cpfs01/shared/opendrivelab/nuscenes/',
        ann_file=
        '/cpfs01/shared/opendrivelab/nuscenes/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(type='CustomLoadMultiViewImageFromFiles', to_float32=False),
            dict(type='CenterCropResizeMultiViewImage', scale=(384, 640)),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='CustomDefaultFormatBundle'),
            dict(type='Collect', keys=['img'], meta_keys=[])
        ],
        load_interval=1,
        queue_length=5,
        condition_frames=1,
        test_mode=True),
    test=dict(
        type='NuScenesTranslationDatasetEval',
        data_root='/cpfs01/shared/opendrivelab/nuscenes/',
        ann_file=
        '/cpfs01/shared/opendrivelab/nuscenes/nuscenes_infos_temporal_val.pkl',
        gen_image_root=
        '/cpfs01/shared/opendrivelab/opendrivelab_hdd/gaoshenyuan/NUSC_outputs/vista_final_infer_nusc_cond',
        pipeline=[
            dict(type='CustomLoadMultiViewImageFromFiles', to_float32=False),
            dict(type='CenterCropResizeMultiViewImage', scale=(384, 640)),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='CustomDefaultFormatBundle'),
            dict(type='Collect', keys=['img'], meta_keys=[])
        ],
        load_interval=1,
        queue_length=5,
        condition_frames=1,
        test_mode=True),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=0, shuffle=False))
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
total_epochs = 10
evaluation = dict(
    interval=1,
    pipeline=[
        dict(type='CustomLoadMultiViewImageFromFiles', to_float32=False),
        dict(type='CenterCropResizeMultiViewImage', scale=(384, 640)),
        dict(
            type='NormalizeMultiviewImage',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='CustomDefaultFormatBundle'),
        dict(type='Collect', keys=['img'], meta_keys=[])
    ])
runner = dict(type='EpochBasedRunner', max_epochs=10)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/cpfs01/user/gaoshenyuan/code/IDM'
load_from = 'ckpts/XVO_translator.pth'
resume_from = None
workflow = [('train', 1)]
auto_resume = False
gpu_ids = range(0, 1)
