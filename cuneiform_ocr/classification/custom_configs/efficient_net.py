num_classes = 145
#classes = ['ABZ579', 'ABZ13', 'ABZ342', 'ABZ70', 'ABZ461', 'ABZ142', 'ABZ480', 'ABZ1', 'ABZ231', 'ABZ533', 'ABZ449', 'ABZ318', 'ABZ75', 'ABZ61', 'ABZ354', 'ABZ139', 'ABZ381', 'ABZ597', 'ABZ536', 'ABZ308', 'ABZ330', 'ABZ328', 'ABZ86', 'ABZ15', 'ABZ214', 'ABZ545', 'ABZ73', 'ABZ295', 'ABZ55', 'ABZ335', 'ABZ371', 'ABZ151', 'ABZ457', 'ABZ537', 'ABZ69', 'ABZ353', 'ABZ68', 'ABZ5', 'ABZ296', 'ABZ84', 'ABZ366', 'ABZ411', 'ABZ396', 'ABZ206', 'ABZ58', 'ABZ324', 'ABZ376', 'ABZ99', 'ABZ384', 'ABZ59', 'ABZ532', 'ABZ334', 'ABZ589', 'ABZ383', 'ABZ343', 'ABZ586', 'ABZ399', 'ABZ74', 'ABZ211', 'ABZ145', 'ABZ7', 'ABZ212', 'ABZ78', 'ABZ367', 'ABZ38', 'ABZ319', 'ABZ85', 'ABZ115', 'ABZ322', 'ABZ97', 'ABZ144', 'ABZ112', 'ABZ427', 'ABZ207', 'ABZ60', 'ABZ79', 'ABZ80', 'ABZ232', 'ABZ142a', 'ABZ312', 'ABZ52', 'ABZ331', 'ABZ128', 'ABZ314', 'ABZ535', 'ABZ575', 'ABZ134', 'ABZ465', 'ABZ167', 'ABZ172', 'ABZ339', 'ABZ6', 'ABZ331e+152i', 'ABZ306', 'ABZ12', 'ABZ2', 'ABZ148', 'ABZ397', 'ABZ554', 'ABZ570', 'ABZ441', 'ABZ147', 'ABZ472', 'ABZ230', 'ABZ440', 'ABZ104', 'ABZ595', 'ABZ455', 'ABZ313', 'ABZ298', 'ABZ62', 'ABZ412', 'ABZ468', 'ABZ101', 'ABZ111', 'ABZ483', 'ABZ538', 'ABZ471', 'ABZ87', 'ABZ143', 'ABZ565', 'ABZ152', 'ABZ205', 'ABZ72', 'ABZ406', 'ABZ138', 'ABZ50', 'ABZ401', 'ABZ307', 'ABZ126', 'ABZ124', 'ABZ164', 'ABZ529', 'ABZ559', 'ABZ94', 'ABZ56', 'ABZ437', 'ABZ393', 'ABZ398']
classes = ['ABZ579', 'ABZ13', 'ABZ342', 'ABZ70', 'ABZ480', 'ABZ461', 'ABZ142', 'ABZ1', 'ABZ533', 'ABZ597', 'ABZ449', 'ABZ231', 'ABZ318', 'ABZ381', 'ABZ61', 'ABZ75', 'ABZ354', 'ABZ139', 'ABZ536', 'ABZ330', 'ABZ308', 'ABZ86', 'ABZ328', 'ABZ545', 'ABZ15', 'ABZ69', 'ABZ295', 'ABZ214', 'ABZ73', 'ABZ296', 'ABZ55', 'ABZ68', 'ABZ537', 'ABZ371', 'ABZ457', 'ABZ151', 'ABZ335', 'ABZ411', 'ABZ366', 'ABZ353', 'ABZ5', 'ABZ84', 'ABZ396', 'ABZ206', 'ABZ324', 'ABZ384', 'ABZ58', 'ABZ376', 'ABZ99', 'ABZ532', 'ABZ74', 'ABZ383', 'ABZ59', 'ABZ334', 'ABZ145', 'ABZ586', 'ABZ589', 'ABZ343', 'ABZ211', 'ABZ7', 'ABZ399', 'ABZ212', 'ABZ78', 'ABZ367', 'ABZ322', 'ABZ60', 'ABZ115', 'ABZ38', 'ABZ319', 'ABZ207', 'ABZ112', 'ABZ85', 'ABZ52', 'ABZ97', 'ABZ144', 'ABZ80', 'ABZ427', 'ABZ79', 'ABZ142a', 'ABZ232', 'ABZ167', 'ABZ535', 'ABZ312', 'ABZ314', 'ABZ331', 'ABZ172', 'ABZ575', 'ABZ306', 'ABZ6', 'ABZ465', 'ABZ339', 'ABZ128', 'ABZ2', 'ABZ401', 'ABZ12', 'ABZ554', 'ABZ595', 'ABZ331e+152i', 'ABZ147', 'ABZ134', 'ABZ397', 'ABZ230', 'ABZ148', 'ABZ104', 'ABZ570', 'ABZ471', 'ABZ440', 'ABZ412', 'ABZ472', 'ABZ393', 'ABZ313', 'ABZ441', 'ABZ62', 'ABZ101', 'ABZ468', 'ABZ111', 'ABZ538', 'ABZ455', 'ABZ298', 'ABZ483', 'ABZ87', 'ABZ205', 'ABZ72', 'ABZ50', 'ABZ143', 'ABZ565', 'ABZ152', 'ABZ307', 'ABZ138', 'ABZ406', 'ABZ94', 'ABZ470', 'ABZ191', 'ABZ9', 'ABZ164', 'ABZ124', 'ABZ398', 'ABZ126', 'ABZ529', 'ABZ437', 'ABZ559', 'ABZ56', 'ABZ10', 'ABZ131', 'ABZ332']

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))
param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=True,
    milestones=[100, 150, 200, 250],
    gamma=0.1)
train_cfg = dict(by_epoch=True, max_epochs=500, val_interval=500)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(base_batch_size=128)
default_scope = 'mmcls'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='CustomLoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=50),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False),
    sync_buffer=dict(type='SyncBuffersHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ClsVisualizer', vis_backends=[dict(type='LocalVisBackend')])
log_level = 'INFO'
load_from = '../../checkpoints/efficientnet-b0_3rdparty_8xb32-aa-advprop_in1k_20220119-26434485.pth'
resume = False
randomness = dict(seed=None, deterministic=False)

model = dict(
    type='ImageClassifier',
    backbone=dict(type='EfficientNet', arch='b0'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=num_classes,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook', cumulative_iters=4)
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    mean=[86.65888836888392, 67.92744567921709, 53.78325960605914],
    std=[68.98970994105028, 57.20489382979894, 48.230552014910586],
    to_rgb=True)
train_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=32,
    num_workers=12,
    dataset=dict(
        type='ClassBalancedDataset',
        dataset=dict(
            type='CustomDataset',
            data_prefix='data/ebl/train_set/train_set',
            classes=classes,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='Resize', scale=380),
                dict(type='RandomFlip', prob=0.25, direction='horizontal'),
                dict(type='RandomFlip', prob=0.25, direction='vertical'),
                dict(type='PackClsInputs')
            ]),
        oversample_thr=0.001),
    sampler=dict(type='DefaultSampler', shuffle=True))

test_dataloader = dict(
    pin_memory=True,
    collate_fn=dict(type='default_collate'),
    persistent_workers=True,
    batch_size=32,
    num_workers=12,
    dataset=dict(
        type='CustomDataset',
        data_prefix='data/ebl/test_set/test_set',
        classes=classes,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=380),
            dict(type='PackClsInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))

test_evaluator = [
    dict(type='Accuracy', topk=(1, 2, 3, 5)),
    dict(type='SingleLabelMetric', items=['precision', 'recall']),
    dict(type='AveragePrecision'),
    dict(type='MultiLabelMetric', average='macro'),
    dict(type='MultiLabelMetric', average='micro')
]

val_dataloader = test_dataloader
val_evaluator = test_evaluator
custom_hooks = [dict(type='CustomTensorboardLoggerHook', by_epoch=True)]
launcher = 'none'
work_dir = 'logs/final1'
