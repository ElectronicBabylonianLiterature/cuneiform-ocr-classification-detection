_base_ = [
    "../configs/textdet/textsnake/textsnake_resnet50-oclip_fpn-unet_1200e_icdar2015.py",
]
load_from = "../checkpoints/textsnake_resnet50-oclip_fpn-unet_1200e_ctw1500_20221101_134814-a216e5b2.pth"

custom_hooks = [dict(type="CustomTensorboardLoggerHook", by_epoch=True)]

_base_.train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=5000, val_interval=100)
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
