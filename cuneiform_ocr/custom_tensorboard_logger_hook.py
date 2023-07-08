# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Optional, Union, Sequence

from mmengine import HOOKS, digit_version
from mmengine.dist import master_only
from mmengine.hooks import LoggerHook
from mmengine.utils.dl_utils import TORCH_VERSION


@HOOKS.register_module()
class CustomTensorboardLoggerHook(LoggerHook):
    """Class to log metrics to Tensorboard.

    Args:
        log_dir (string): Save directory location. Default: None. If default
            values are used, directory location is ``runner.work_dir``/tf_logs.
        interval (int): Logging interval (every k iterations). Default: True.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`. Default: True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default: False.
        by_epoch (bool): Whether EpochBasedRunner is used. Default: True.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        interval: int = 10,
        ignore_last: bool = True,
        reset_flag: bool = False,
        by_epoch: bool = True,
    ):
        super().__init__(
            interval=interval, ignore_last=ignore_last, log_metric_by_epoch=by_epoch
        )
        self.log_dir = log_dir

    def before_run(self, runner) -> None:
        super().before_run(runner)
        if TORCH_VERSION == "parrots" or digit_version(TORCH_VERSION) < digit_version(
            "1.1"
        ):
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError(
                    "Please install tensorboardX to use " "TensorboardLoggerHook."
                )
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    "the dependencies to use torch.utils.tensorboard "
                    "(applicable to PyTorch 1.1 or higher)"
                )

        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, "tf_logs")
        self.writer = SummaryWriter(self.log_dir)

    def after_train_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: Optional[Union[dict, tuple, list]] = None,
        outputs: Optional[dict] = None,
    ) -> None:
        tags, _ = runner.log_processor.get_log_after_iter(runner, batch_idx, "train")
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, runner.iter + 1)
            else:
                self.writer.add_scalar(tag, val, runner.iter + 1)

    def after_val_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: Optional[Union[dict, tuple, list]] = None,
        outputs: Optional[Sequence] = None,
    ) -> None:
        if self.every_n_inner_iters(batch_idx, self.interval):
            tags, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, "val"
            )
            for tag, val in tags.items():
                if isinstance(val, str):
                    self.writer.add_text(tag, val, runner.iter + 1)
                else:
                    self.writer.add_scalar(tag, val, runner.iter + 1)

    def after_run(self, runner) -> None:
        self.writer.close()
