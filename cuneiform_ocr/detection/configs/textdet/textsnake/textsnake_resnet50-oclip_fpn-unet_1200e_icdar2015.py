_base_ = [
    "textsnake_resnet50_fpn-unet_1200e_icdar2015.py",
]

_base_.model.backbone = dict(
    type="CLIPResNet",
    init_cfg=dict(
        type="Pretrained",
        checkpoint="https://download.openmmlab.com/"
        "mmocr/backbone/resnet50-oclip-7ba0c533.pth",
    ),
)
