train_pipeline_default = [
    dict(
        type="LoadImageFromFile",
        file_client_args=dict(backend="disk"),
        color_type="color_ignore_orientation",
    ),
    dict(type="LoadOCRAnnotations", with_polygon=True, with_bbox=True, with_label=True),
    dict(
        type="RandomResize", scale=(800, 800), ratio_range=(0.75, 2.5), keep_ratio=True
    ),
    dict(type="TextDetRandomCropFlip", crop_ratio=0.5, iter_num=1, min_area_ratio=0.2),
    dict(
        type="RandomApply",
        transforms=[dict(type="RandomCrop", min_side_ratio=0.3)],
        prob=0.8,
    ),
    dict(
        type="RandomApply",
        transforms=[
            dict(
                type="RandomRotate",
                max_angle=30,
                pad_with_fixed_color=False,
                use_canvas=True,
            )
        ],
        prob=0.5,
    ),
    dict(
        type="RandomChoice",
        transforms=[
            [
                {"type": "Resize", "scale": 800, "keep_ratio": True},
                {"type": "SourceImagePad", "target_scale": 800},
            ],
            {"type": "Resize", "scale": 800, "keep_ratio": False},
        ],
        prob=[0.6, 0.4],
    ),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(
        type="TorchVisionWrapper",
        op="ColorJitter",
        brightness=0.12549019607843137,
        saturation=0.5,
        contrast=0.5,
    ),
    dict(
        type="PackTextDetInputs",
        meta_keys=("img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]


train_pipeline_enhanced1 = [
    dict(
        type="LoadImageFromFile",
        file_client_args=dict(backend="disk"),
        color_type="color_ignore_orientation",
    ),
    dict(type="LoadOCRAnnotations", with_polygon=True, with_bbox=True, with_label=True),
    dict(type="FixInvalidPolygon", min_poly_points=4),
    dict(
        type="RandomResize", scale=(800, 800), ratio_range=(0.75, 2.5), keep_ratio=True
    ),
    dict(type="FixInvalidPolygon", min_poly_points=4),
    dict(type="TextDetRandomCropFlip", crop_ratio=0.5, iter_num=1, min_area_ratio=0.2),
    dict(
        type="RandomApply",
        transforms=[dict(type="RandomCrop", min_side_ratio=0.3)],
        prob=0.8,
    ),
    dict(type="FixInvalidPolygon", min_poly_points=4),
    dict(
        type="RandomApply",
        transforms=[
            dict(
                type="RandomRotate",
                max_angle=35,
                pad_with_fixed_color=True,
                use_canvas=True,
            )
        ],
        prob=0.6,
    ),
    dict(type="FixInvalidPolygon", min_poly_points=4),
    dict(
        type="RandomChoice",
        transforms=[
            [
                {"type": "Resize", "scale": 800, "keep_ratio": True},
                {"type": "Pad", "size": (800, 800)},
            ],
            {"type": "Resize", "scale": 800, "keep_ratio": False},
        ],
        prob=[0.6, 0.4],
    ),
    dict(type="FixInvalidPolygon", min_poly_points=4),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="FixInvalidPolygon", min_poly_points=4),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    dict(type="FixInvalidPolygon", min_poly_points=4),
    dict(
        type="RandomApply",
        transforms=[
            dict(type="TorchVisionWrapper", op="ElasticTransform", alpha=75.0),
        ],
        prob=1/3,
    ),
    dict(type="FixInvalidPolygon", min_poly_points=4),
    dict(
        type="RandomApply",
        transforms=[
            dict(
                type="RandomChoice",
                transforms=[
                    dict(
                        type="TorchVisionWrapper",
                        op="RandomAdjustSharpness",
                        sharpness_factor=0,
                    ),
                    dict(
                        type="TorchVisionWrapper",
                        op="RandomAdjustSharpness",
                        sharpness_factor=60,
                    ),
                    dict(
                        type="TorchVisionWrapper",
                        op="RandomAdjustSharpness",
                        sharpness_factor=90,
                    ),
                ],
                prob=[1/3] * 3,
            ),
        ],
        prob=0.75,
    ),
    dict(type="FixInvalidPolygon", min_poly_points=4),
    dict(
        type="TorchVisionWrapper",
        op="ColorJitter",
        brightness=0.15,
        saturation=0.5,
        contrast=0.3,
    ),
    dict(
        type="RandomApply",
        transforms=[
            dict(
                type="RandomChoice",
                transforms=[
                    dict(type="TorchVisionWrapper", op="RandomEqualize"),
                    dict(type="TorchVisionWrapper", op="RandomAutocontrast"),
                ],
                prob=[1 / 2, 1 / 2],
            ),
        ],
        prob=0.8,
    ),
    dict(type="FixInvalidPolygon", min_poly_points=4),
    dict(
        type="PackTextDetInputs",
        meta_keys=("img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_pipeline_enhanced = [
    dict(
        type="LoadImageFromFile",
        file_client_args=dict(backend="disk"),
        color_type="color_ignore_orientation",
    ),
    dict(type="LoadOCRAnnotations", with_polygon=True, with_bbox=True, with_label=True),
    dict(type="FixInvalidPolygon", min_poly_points=4),
    dict(
        type="RandomResize", scale=(800, 800), ratio_range=(0.75, 2.5), keep_ratio=True
    ),
    dict(type="TextDetRandomCropFlip", crop_ratio=0.5, iter_num=1, min_area_ratio=0.2),
    dict(
        type="RandomApply",
        transforms=[dict(type="RandomCrop", min_side_ratio=0.3)],
        prob=0.8,
    ),
    dict(
        type="RandomApply",
        transforms=[
            dict(
                type="RandomRotate",
                max_angle=35,
                pad_with_fixed_color=True,
                use_canvas=True,
            )
        ],
        prob=0.6,
    ),
    dict(
        type="RandomChoice",
        transforms=[
            [
                {"type": "Resize", "scale": 800, "keep_ratio": True},
                {"type": "Pad", "size": (800, 800)},
            ],
            {"type": "Resize", "scale": 800, "keep_ratio": False},
        ],
        prob=[0.6, 0.4],
    ),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    dict(
        type="RandomApply",
        transforms=[
            dict(type="TorchVisionWrapper", op="ElasticTransform", alpha=75.0),
        ],
        prob=1/3,
    ),
    dict(
        type="RandomApply",
        transforms=[
            dict(
                type="RandomChoice",
                transforms=[
                    dict(
                        type="TorchVisionWrapper",
                        op="RandomAdjustSharpness",
                        sharpness_factor=0,
                    ),
                    dict(
                        type="TorchVisionWrapper",
                        op="RandomAdjustSharpness",
                        sharpness_factor=60,
                    ),
                    dict(
                        type="TorchVisionWrapper",
                        op="RandomAdjustSharpness",
                        sharpness_factor=90,
                    ),
                ],
                prob=[1/3] * 3,
            ),
        ],
        prob=0.75,
    ),
    dict(
        type="TorchVisionWrapper",
        op="ColorJitter",
        brightness=0.15,
        saturation=0.5,
        contrast=0.3,
    ),
    dict(
        type="RandomApply",
        transforms=[
            dict(
                type="RandomChoice",
                transforms=[
                    dict(type="TorchVisionWrapper", op="RandomEqualize"),
                    dict(type="TorchVisionWrapper", op="RandomAutocontrast"),
                ],
                prob=[1 / 2, 1 / 2],
            ),
        ],
        prob=0.8,
    ),
    dict(type="FixInvalidPolygon", min_poly_points=4),
    dict(
        type="PackTextDetInputs",
        meta_keys=("img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]