auto_scale_lr = dict(base_batch_size=1024)
bgr_mean = [
    103.53,
    116.28,
    123.675,
]
bgr_std = [
    57.375,
    57.12,
    58.395,
]
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    num_classes=4,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
dataset_type = 'CustomDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1, max_keep_ckpts=10, save_best='auto',
        type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = '/home/jhkim/mmlab/mmpretrain/projects/phamcle_phase1_base2_invert/best_accuracy_top1_epoch_66.pth'
log_level = 'INFO'
model = dict(
    backbone=dict(
        arch='base',
        drop_path_rate=0.2,
        img_size=384,
        init_cfg=dict(
            checkpoint=
            'projects/KDCA/swinv2_base/best_accuracy_top1_epoch_47.pth',
            prefix='backbone.',
            type='Pretrained'),
        pad_small_map=True,
        pretrained_window_sizes=[
            24,
            24,
            24,
            12,
        ],
        type='SwinTransformerV2',
        window_size=[
            24,
            24,
            24,
            12,
        ]),
    head=dict(
        cal_acc=False,
        in_channels=1024,
        init_cfg=None,
        loss=dict(
            alpha=0.4,
            gamma=2.5,
            loss_weight=1.0,
            reduction='mean',
            type='FocalLoss'),
        num_classes=4,
        type='LinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.0001,
        type='AdamW',
        weight_decay=0.005),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        custom_keys=dict({
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
        flat_decay_mult=0.0,
        norm_decay_mult=0.0))
param_scheduler = [
    dict(
        by_epoch=True,
        convert_to_iter_based=True,
        end=10,
        start_factor=0.0005,
        type='LinearLR'),
    dict(begin=10, by_epoch=True, eta_min=1e-05, type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=1,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_prefix='test',
        data_root='data/phamcle_phase1_base2_invert',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=384, type='Resize'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset'),
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    topk=(
        1,
        2,
    ), type='Accuracy')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=384, type='Resize'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
train_dataloader = dict(
    batch_size=8,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_prefix='train',
        data_root='data/phamcle_phase1_base2_invert',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                interpolation='bicubic',
                scale=384,
                type='RandomResizedCrop'),
            dict(to_rgb=True, type='NumpyToPIL'),
            dict(
                distortion_scale=0.5,
                p=0.5,
                type='torchvision/RandomPerspective'),
            dict(
                degrees=180,
                scale=[
                    0.85,
                    1.15,
                ],
                type='torchvision/RandomAffine'),
            dict(to_bgr=True, type='PILToNumpy'),
            dict(
                direction=[
                    'horizontal',
                    'vertical',
                    'diagonal',
                ],
                prob=0.5,
                type='RandomFlip'),
            dict(
                hparams=dict(
                    interpolation='bicubic', pad_val=[
                        104,
                        116,
                        124,
                    ]),
                magnitude_level=9,
                magnitude_std=0.5,
                num_policies=2,
                policies=[
                    dict(type='AutoContrast'),
                    dict(type='Equalize'),
                    dict(type='Invert'),
                    dict(magnitude_range=(
                        4,
                        0,
                    ), type='Posterize'),
                    dict(magnitude_range=(
                        256,
                        0,
                    ), type='Solarize'),
                    dict(magnitude_range=(
                        0,
                        110,
                    ), type='SolarizeAdd'),
                    dict(magnitude_range=(
                        0,
                        0.9,
                    ), type='ColorTransform'),
                    dict(magnitude_range=(
                        0,
                        0.9,
                    ), type='Contrast'),
                    dict(magnitude_range=(
                        0,
                        0.9,
                    ), type='Brightness'),
                    dict(magnitude_range=(
                        0,
                        0.9,
                    ), type='Sharpness'),
                    dict(
                        magnitude_range=(
                            0.1,
                            2.0,
                        ),
                        magnitude_std='inf',
                        prob=1.0,
                        type='GaussianBlur'),
                    dict(
                        magnitude_range=(
                            50,
                            100,
                        ),
                        magnitude_std='inf',
                        pad_val=0,
                        type='Cutout'),
                    dict(
                        direction='horizontal',
                        magnitude_range=(
                            0,
                            0.3,
                        ),
                        type='Shear'),
                    dict(
                        direction='vertical',
                        magnitude_range=(
                            0,
                            0.3,
                        ),
                        type='Shear'),
                    dict(
                        direction='horizontal',
                        magnitude_range=(
                            0,
                            0.45,
                        ),
                        type='Translate'),
                    dict(
                        direction='vertical',
                        magnitude_range=(
                            0,
                            0.45,
                        ),
                        type='Translate'),
                ],
                total_level=10,
                type='RandAugment'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        backend='pillow',
        interpolation='bicubic',
        scale=384,
        type='RandomResizedCrop'),
    dict(to_rgb=True, type='NumpyToPIL'),
    dict(distortion_scale=0.5, p=0.5, type='torchvision/RandomPerspective'),
    dict(degrees=180, scale=[
        0.85,
        1.15,
    ], type='torchvision/RandomAffine'),
    dict(to_bgr=True, type='PILToNumpy'),
    dict(
        direction=[
            'horizontal',
            'vertical',
            'diagonal',
        ],
        prob=0.5,
        type='RandomFlip'),
    dict(
        hparams=dict(interpolation='bicubic', pad_val=[
            104,
            116,
            124,
        ]),
        magnitude_level=9,
        magnitude_std=0.5,
        num_policies=2,
        policies=[
            dict(type='AutoContrast'),
            dict(type='Equalize'),
            dict(type='Invert'),
            dict(magnitude_range=(
                4,
                0,
            ), type='Posterize'),
            dict(magnitude_range=(
                256,
                0,
            ), type='Solarize'),
            dict(magnitude_range=(
                0,
                110,
            ), type='SolarizeAdd'),
            dict(magnitude_range=(
                0,
                0.9,
            ), type='ColorTransform'),
            dict(magnitude_range=(
                0,
                0.9,
            ), type='Contrast'),
            dict(magnitude_range=(
                0,
                0.9,
            ), type='Brightness'),
            dict(magnitude_range=(
                0,
                0.9,
            ), type='Sharpness'),
            dict(
                magnitude_range=(
                    0.1,
                    2.0,
                ),
                magnitude_std='inf',
                prob=1.0,
                type='GaussianBlur'),
            dict(
                magnitude_range=(
                    50,
                    100,
                ),
                magnitude_std='inf',
                pad_val=0,
                type='Cutout'),
            dict(
                direction='horizontal',
                magnitude_range=(
                    0,
                    0.3,
                ),
                type='Shear'),
            dict(
                direction='vertical', magnitude_range=(
                    0,
                    0.3,
                ), type='Shear'),
            dict(
                direction='horizontal',
                magnitude_range=(
                    0,
                    0.45,
                ),
                type='Translate'),
            dict(
                direction='vertical',
                magnitude_range=(
                    0,
                    0.45,
                ),
                type='Translate'),
        ],
        total_level=10,
        type='RandAugment'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=1,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_prefix='validation',
        data_root='data/phamcle_phase1_base2_invert',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=384, type='Resize'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset'),
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    topk=(
        1,
        2,
    ), type='Accuracy')
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=384, type='Resize'),
    dict(type='PackInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'projects/phamcle_phase1_base2_invert'
