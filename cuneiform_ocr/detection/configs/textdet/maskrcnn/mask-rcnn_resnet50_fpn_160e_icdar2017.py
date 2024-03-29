_base_ = [
    "mask-rcnn_resnet50_fpn_160e_icdar2015.py",
    "../_base_/datasets/icdar2017.py",
]

icdar2017_textdet_train = _base_.icdar2017_textdet_train
icdar2017_textdet_test = _base_.icdar2017_textdet_test
# use the same pipeline as icdar2015
icdar2017_textdet_train.pipeline = _base_.train_pipeline
icdar2017_textdet_test.pipeline = _base_.test_pipeline

train_dataloader = dict(dataset=icdar2017_textdet_train)
val_dataloader = dict(dataset=icdar2017_textdet_test)
test_dataloader = val_dataloader
