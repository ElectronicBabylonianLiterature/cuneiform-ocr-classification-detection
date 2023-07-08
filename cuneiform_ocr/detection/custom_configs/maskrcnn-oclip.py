_base_ = [
    "../configs/textdet/maskrcnn/mask-rcnn_resnet50-oclip_fpn_160e_icdar2015.py",
]


load_from = "../../checkpoints/detection/fcenet_resnet50-dcnv2_fpn_1500e_ctw1500_20220825_221510-4d705392.pth"

custom_hooks = [dict(type="CustomTensorboardLoggerHook", by_epoch=True)]
_base_.train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=1000, val_interval=100)
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
