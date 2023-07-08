_base_ = [
    "../configs/textdet/fcenet/fcenet_resnet50-oclip_fpn_1500e_icdar2015.py",
]

_base_.train_dataloader.batch_size = 8
_base_.auto_scale_lr = dict(base_batch_size=8)
optimizer_config = {"type": "GradientCumulativeOptimizerHook", "cumulative_iters": 2}

load_from = "../checkpoints/fcenet_resnet50-oclip_fpn_1500e_icdar2015_20221101_150145-5a6fc412.pth"

custom_hooks = [dict(type="CustomTensorboardLoggerHook", by_epoch=True)]
_base_.train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=3500, val_interval=100)
_base_.default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="CustomLoggerHook", interval=1),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=100),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    sync_buffer=dict(type="SyncBuffersHook"),
    visualization=dict(
        type="VisualizationHook",
        interval=1,
        enable=True,
        show=False,
        draw_gt=True,
        draw_pred=True,
    ),
)
_base_.visualizer = dict(
    type="TextDetLocalVisualizer",
    name="visualizer",
    vis_backends=[dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")],
)
