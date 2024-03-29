# Note: This schedule config serves as a base config for other schedules.
# Users would have to at least fill in "max_epochs" and "val_interval"
# in order to use this config in their experiments.

# optimizer
optim_wrapper = dict(type="OptimWrapper", optimizer=dict(type="Adam", lr=3e-4))
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=None, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
# learning policy
param_scheduler = [
    dict(type="ConstantLR", factor=1.0),
]
