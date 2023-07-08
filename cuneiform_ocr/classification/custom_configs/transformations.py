dict(
    type="AutoAugment",
    policies="imagenet",
    hparams=dict(pad_val=[round(x) for x in bgr_mean], interpolation="bicubic"),
),
dict(
    type="RandAugment",
    policies="timm_increasing",
    num_policies=2,
    total_level=10,
    magnitude_level=9,
    magnitude_std=0.5,
    hparams=dict(pad_val=[round(x) for x in bgr_mean], interpolation="bicubic"),
),
dict(
    type="RandomErasing",
    erase_prob=0.25,
    mode="rand",
    min_area_ratio=0.02,
    max_area_ratio=1 / 3,
    fill_color=bgr_mean,
    fill_std=bgr_std,
),
