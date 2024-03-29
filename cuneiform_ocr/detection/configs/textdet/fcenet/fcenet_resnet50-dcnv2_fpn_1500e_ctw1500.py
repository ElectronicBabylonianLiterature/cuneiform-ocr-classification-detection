_base_ = [
    "_base_fcenet_resnet50-dcnv2_fpn.py",
    "../_base_/datasets/icdar2015.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_sgd_base.py",
]

optim_wrapper = dict(optimizer=dict(lr=1e-3, weight_decay=5e-4))
train_cfg = dict(max_epochs=1500)
# learning policy
param_scheduler = [
    dict(type="PolyLR", power=0.9, eta_min=1e-7, end=1500),
]

file_client_args = dict(backend="disk")
# dataset settings
# dataset settings
icdar2015_textdet_train = _base_.icdar2015_textdet_train
icdar2015_textdet_test = _base_.icdar2015_textdet_test
icdar2015_textdet_train.pipeline = _base_.train_pipeline
icdar2015_textdet_test.pipeline = _base_.test_pipeline

# test pipeline for CTW1500
ctw_test_pipeline = [
    dict(
        type="LoadImageFromFile",
        file_client_args=file_client_args,
        color_type="color_ignore_orientation",
    ),
    dict(type="Resize", scale=(1080, 736), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadOCRAnnotations", with_polygon=True, with_bbox=True, with_label=True),
    dict(
        type="PackTextDetInputs",
        meta_keys=("img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]


train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=icdar2015_textdet_train,
)

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=icdar2015_textdet_test,
)

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=8)
