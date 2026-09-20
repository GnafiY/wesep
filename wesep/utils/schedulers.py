# Copyright (c) 2021 Shuai Wang (wsstriving@gmail.com)
#               2021 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2022 Hongji Wang (jijijiang77@gmail.com)
#               2026 Ke Zhang (kylezhang1118@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Learning-rate scheduler helpers used by wesep/bin/train.py.

The trainer calls these public methods at fixed points:
  - step_batch(): before each optimizer update,
  - step_validation(): after validation loss is available,
  - step_epoch(): at the end of an epoch.

Each method returns a SchedulerResult so the trainer can log lr changes or
stop early without knowing the scheduler-specific rules.
"""

import math
from dataclasses import dataclass

SUPPORTED_SCHEDULERS = (
    "ExponentialDecrease",
    "PlateauHalf",
    "StepDecay",
    "TriAngular2",
    "none",
    "NoOpScheduler",
)


@dataclass
class SchedulerResult:
    """Decision returned by one scheduler step.

    The trainer consumes this instead of checking scheduler internals.
    """

    lr_changed: bool = False
    should_stop: bool = False
    is_best: bool = False
    in_warmup: bool = False
    message: str = ""


class WarmupController:
    """Optional linear warmup shared by all LR schedulers.

    Args:
        optimizer: Optimizer whose ``param_group["lr"]`` values are updated.
        base_lrs: Target lrs at the end of warmup, one per param group.
        epoch_iter: Number of optimizer steps in one epoch.
        config: Optional dict. ``by`` may be ``"step"`` or ``"epoch"``.

    Returns:
        ``step`` returns a SchedulerResult describing whether lr changed.
    """

    def __init__(self, optimizer, base_lrs, epoch_iter, config=None):
        config = config or {}
        self.optimizer = optimizer
        self.base_lrs = list(base_lrs)
        self.epoch_iter = int(epoch_iter)
        self.enabled = bool(config) and config.get("enabled", True)
        self.by = config.get("by", "epoch" if "epochs" in config else "step")
        self.steps = int(config.get("steps", 0) or 0)
        self.epochs = int(config.get("epochs", 0) or 0)
        self.start_factor = float(config.get("start_factor", 0.0))
        self.end_factor = float(config.get("end_factor", 1.0))

        if self.by not in ("step", "epoch"):
            raise ValueError("warmup.by must be 'step' or 'epoch'")
        self._refresh_lengths()

        self.enabled = self.enabled and self.steps > 0

    def _refresh_lengths(self):
        """Keep step-based and epoch-based warmup lengths consistent."""
        if self.by == "epoch":
            self.steps = self.epochs * self.epoch_iter
        else:
            self.epochs = (math.ceil(self.steps / self.epoch_iter)
                           if self.epoch_iter > 0 else 0)

    def is_active(self, global_step):
        """Return true while warmup owns the lr."""
        return self.enabled and global_step < self.steps

    def skip_validation(self, epoch, global_step):
        """Return true when validation/epoch schedulers should wait."""
        if not self.enabled:
            return False
        if self.by == "epoch":
            return epoch <= self.epochs
        return global_step < self.steps

    def effective_step(self, global_step):
        return max(global_step - self.steps, 0)

    def effective_epoch(self, epoch):
        return max(epoch - self.epochs, 0)

    def step(self, global_step):
        """Linearly move lr from start_factor to end_factor of base_lrs."""
        if not self.is_active(global_step):
            return SchedulerResult()

        progress = min(float(global_step + 1) / float(self.steps), 1.0)
        factor = self.start_factor + progress * (self.end_factor -
                                                 self.start_factor)
        for base_lr, param_group in zip(self.base_lrs,
                                        self.optimizer.param_groups):
            param_group["lr"] = base_lr * factor
        return SchedulerResult(lr_changed=True, in_warmup=True)

    def state_dict(self):
        """Save only plain values; optimizer links are rebuilt at runtime."""
        return {
            "enabled": self.enabled,
            "by": self.by,
            "steps": self.steps,
            "epochs": self.epochs,
            "start_factor": self.start_factor,
            "end_factor": self.end_factor,
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._refresh_lengths()

    def set_epoch_iter(self, epoch_iter):
        self.epoch_iter = int(epoch_iter)
        self._refresh_lengths()


class LRSchedulerBase:
    """Base class that routes train.py callbacks to scheduler implementations.

    Subclasses usually implement one of ``_step_step``, ``_step_epoch``, or
    ``_step_validation``. Public step methods always return SchedulerResult.

    Args:
        optimizer: Optimizer whose lr should be controlled.
        num_epochs: Total planned training epochs for progress-based schedules.
        epoch_iter: Number of optimizer steps in one epoch.
        interval: Which callback should update lr.
        warmup: Optional WarmupController config dict.
    """

    valid_intervals = ("step", "epoch", "validation", "none")

    def __init__(self,
                 optimizer,
                 num_epochs,
                 epoch_iter,
                 interval="step",
                 warmup=None):
        if interval not in self.valid_intervals:
            raise ValueError(f"Invalid scheduler interval: {interval}")

        self.optimizer = optimizer
        self.num_epochs = int(num_epochs)
        self.epoch_iter = int(epoch_iter)
        self.interval = interval
        self.base_lrs = [
            float(group["lr"]) for group in optimizer.param_groups
        ]
        self.warmup = WarmupController(optimizer, self.base_lrs,
                                       self.epoch_iter, warmup)
        self.current_step = 0
        self.current_epoch = 0

    def __repr__(self):
        return f"{self.__class__.__name__}(interval={self.interval})"

    def get_lrs(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def set_lrs(self, lrs):
        for lr, param_group in zip(lrs, self.optimizer.param_groups):
            param_group["lr"] = float(lr)

    def main_total_steps(self):
        return max(self.num_epochs * self.epoch_iter - self.warmup.steps, 1)

    def main_total_epochs(self):
        return max(self.num_epochs - self.warmup.epochs, 1)

    def step_batch(self, global_step, epoch=None):
        self.current_step = global_step
        # Warmup takes priority over the main scheduler until it finishes.
        if self.warmup.is_active(global_step):
            return self.warmup.step(global_step)
        if self.interval != "step":
            return SchedulerResult()
        return self._step_step(self.warmup.effective_step(global_step), epoch)

    def step_epoch(self, epoch, global_step=None):
        self.current_epoch = epoch
        if self.interval != "epoch":
            return SchedulerResult()
        # Epoch/validation schedulers should not react to warmup epochs.
        if self.warmup.skip_validation(epoch, global_step or 0):
            return SchedulerResult(in_warmup=True)
        effective_epoch = max(self.warmup.effective_epoch(epoch) - 1, 0)
        return self._step_epoch(effective_epoch, global_step)

    def step_validation(self, metric, epoch, global_step=None):
        if self.interval != "validation":
            return SchedulerResult()
        # A validation scheduler should see the first post-warmup validation.
        if self.warmup.skip_validation(epoch, global_step or 0):
            return SchedulerResult(in_warmup=True)
        return self._step_validation(metric, epoch, global_step)

    def _step_step(self, current_step, epoch=None):
        return SchedulerResult()

    def _step_epoch(self, current_epoch, global_step=None):
        return SchedulerResult()

    def _step_validation(self, metric, epoch, global_step=None):
        return SchedulerResult()

    def state_dict(self):
        # The optimizer and WarmupController object are runtime links, so only
        # store simple scheduler state plus warmup's own small state dict.
        state = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "warmup")
        }
        state["warmup"] = self.warmup.state_dict()
        return state

    def load_state_dict(self, state_dict):
        # Checkpoints restore historical counters, but the dataloader shape
        # belongs to the current run. This matters when resuming with a changed
        # batch size or GPU count.
        num_epochs = self.num_epochs
        epoch_iter = self.epoch_iter
        state_dict = dict(state_dict)
        warmup_state = state_dict.pop("warmup", None)
        self.__dict__.update(state_dict)
        if warmup_state is not None:
            self.warmup.load_state_dict(warmup_state)
        self.set_runtime_config(num_epochs=num_epochs, epoch_iter=epoch_iter)

    def set_runtime_config(self, num_epochs=None, epoch_iter=None):
        """Apply run-specific values that should not come from checkpoints."""
        if num_epochs is not None:
            self.num_epochs = int(num_epochs)
        if epoch_iter is not None:
            self.epoch_iter = int(epoch_iter)
        self.warmup.base_lrs = list(self.base_lrs)
        self.warmup.set_epoch_iter(self.epoch_iter)


class ExponentialDecrease(LRSchedulerBase):
    """Exponentially decay lr from the optimizer base lr to a final lr.

    This can update every optimizer step or once per epoch, depending on
    ``interval``. ``final_lr`` may be a scalar or one value per param group.
    """

    def __init__(self,
                 optimizer,
                 num_epochs,
                 epoch_iter,
                 final_lr=None,
                 final_lr_factor=None,
                 interval="step",
                 warmup=None):
        super().__init__(optimizer, num_epochs, epoch_iter, interval, warmup)
        if self.interval not in ("step", "epoch"):
            raise ValueError(
                "ExponentialDecrease only supports interval='step' or "
                "interval='epoch'")
        if final_lr is None and final_lr_factor is None:
            raise ValueError(
                "ExponentialDecrease requires final_lr or final_lr_factor")
        self.final_lr = final_lr
        self.final_lr_factor = final_lr_factor
        self.final_lrs = self._build_final_lrs(final_lr, final_lr_factor)

    def _build_final_lrs(self, final_lr, final_lr_factor):
        """Return final lr values aligned with optimizer param groups."""
        if final_lr is None:
            return [
                base_lr * float(final_lr_factor) for base_lr in self.base_lrs
            ]
        if isinstance(final_lr, (list, tuple)):
            if len(final_lr) != len(self.base_lrs):
                raise ValueError(
                    "final_lr list length must match optimizer param groups")
            return [float(lr) for lr in final_lr]
        return [float(final_lr) for _ in self.base_lrs]

    def _get_lrs(self, current, total):
        """Interpolate in log space; current is clamped to the schedule range."""
        if total <= 1:
            progress = 1.0
        else:
            progress = min(max(current, 0), total - 1) / float(total - 1)
        lrs = []
        for base_lr, final_lr in zip(self.base_lrs, self.final_lrs):
            if base_lr <= 0.0 or final_lr <= 0.0:
                raise ValueError(
                    "ExponentialDecrease requires positive learning rates")
            lrs.append(base_lr *
                       math.exp(progress * math.log(final_lr / base_lr)))
        return lrs

    def _step_step(self, current_step, epoch=None):
        self.set_lrs(self._get_lrs(current_step, self.main_total_steps()))
        return SchedulerResult(lr_changed=True)

    def _step_epoch(self, current_epoch, global_step=None):
        self.set_lrs(self._get_lrs(current_epoch, self.main_total_epochs()))
        return SchedulerResult(lr_changed=True)


class PlateauHalf(LRSchedulerBase):
    """Reduce lr when validation loss does not improve.

    This scheduler only looks at validation loss. It does not roll the model
    back to ``best_checkpoint.pt``; train.py saves that checkpoint separately.
    """

    def __init__(self,
                 optimizer,
                 num_epochs,
                 epoch_iter,
                 patience=3,
                 factor=0.5,
                 min_delta=0.0,
                 min_lr=0.0,
                 early_stop_patience=None,
                 early_stop=False,
                 interval="validation",
                 warmup=None):
        super().__init__(optimizer, num_epochs, epoch_iter, interval, warmup)
        if self.interval != "validation":
            raise ValueError("PlateauHalf only supports interval='validation'")
        if not 0.0 < float(factor) < 1.0:
            raise ValueError("factor must be in (0, 1)")

        self.patience = int(patience)
        self.factor = float(factor)
        self.min_delta = float(min_delta)
        self.min_lr = float(min_lr)
        self.early_stop_patience = (None if early_stop_patience is None else
                                    int(early_stop_patience))
        self.early_stop = bool(early_stop)
        self.best = None
        self.bad_epochs = 0
        self.epochs_since_reduction = 0
        self.num_reductions = 0
        self.should_stop = False

    def _is_better(self, metric):
        """Return true when a smaller validation loss beats the current best."""
        if self.best is None:
            return True
        return metric < self.best - self.min_delta

    def _reduce_lrs(self):
        """Half each param-group lr, respecting min_lr."""
        old_lrs = self.get_lrs()
        new_lrs = [max(lr * self.factor, self.min_lr) for lr in old_lrs]
        changed = any(new_lr < old_lr
                      for old_lr, new_lr in zip(old_lrs, new_lrs))
        if changed:
            self.set_lrs(new_lrs)
            self.num_reductions += 1
            self.epochs_since_reduction = 0
        return changed

    def _step_validation(self, metric, epoch, global_step=None):
        metric = float(metric)

        # A new best validation loss resets all plateau counters.
        if self._is_better(metric):
            self.best = metric
            self.bad_epochs = 0
            self.epochs_since_reduction = 0
            return SchedulerResult(is_best=True)

        # bad_epochs counts since best; epochs_since_reduction controls how
        # often lr is reduced while the metric remains on a plateau.
        self.bad_epochs += 1
        self.epochs_since_reduction += 1

        # Optional hard stop after N bad validation epochs, independent of lr.
        if (self.early_stop and self.early_stop_patience is not None
                and self.bad_epochs >= self.early_stop_patience):
            self.should_stop = True
            return SchedulerResult(
                should_stop=True,
                message=("PlateauHalf early stopping triggered after {} bad "
                         "validation epoch(s)").format(self.bad_epochs),
            )

        if self.epochs_since_reduction < self.patience:
            return SchedulerResult()

        # Reduce lr every patience bad epochs. If lr is already at min_lr, the
        # optional early-stop branch below can end training.
        if self._reduce_lrs():
            return SchedulerResult(
                lr_changed=True,
                message=("PlateauHalf reduced lr to {:.6g} after {} bad "
                         "validation epoch(s)").format(self.get_lr(),
                                                       self.bad_epochs),
            )

        if self.early_stop and self.early_stop_patience is None:
            self.should_stop = True
            return SchedulerResult(
                should_stop=True,
                message=("PlateauHalf early stopping triggered after {} lr "
                         "reduction(s)").format(self.num_reductions),
            )

        self.epochs_since_reduction = 0
        return SchedulerResult()


class StepDecay(LRSchedulerBase):
    """Multiply lr by ``gamma`` every ``step_size`` steps or epochs."""

    def __init__(self,
                 optimizer,
                 num_epochs,
                 epoch_iter,
                 step_size,
                 gamma=0.5,
                 min_lr=0.0,
                 interval="step",
                 warmup=None):
        super().__init__(optimizer, num_epochs, epoch_iter, interval, warmup)
        if self.interval not in ("step", "epoch"):
            raise ValueError(
                "StepDecay only supports interval='step' or interval='epoch'")
        self.step_size = int(step_size)
        self.gamma = float(gamma)
        self.min_lr = float(min_lr)
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")

    def _get_lrs(self, num_completed):
        """Return lrs after ``num_completed`` scheduled units."""
        num_completed = max(num_completed, 0)
        factor = self.gamma**(num_completed // self.step_size)
        return [
            max(base_lr * factor, self.min_lr) for base_lr in self.base_lrs
        ]

    def _step_step(self, current_step, epoch=None):
        self.set_lrs(self._get_lrs(current_step))
        return SchedulerResult(lr_changed=True)

    def _step_epoch(self, current_epoch, global_step=None):
        self.set_lrs(self._get_lrs(current_epoch + 1))
        return SchedulerResult(lr_changed=True)


class TriAngular2(LRSchedulerBase):
    """Cyclical triangular lr schedule with a smaller peak each cycle.

    ``cycle_step`` is measured in epochs. The derived ``cycle_iter`` and
    ``step_size`` are refreshed on resume so batch-size changes are handled.
    """

    def __init__(self,
                 optimizer,
                 num_epochs,
                 epoch_iter,
                 final_lr,
                 interval="step",
                 warmup=None,
                 cycle_step=2,
                 reduce_lr_diff_ratio=0.5):
        super().__init__(optimizer, num_epochs, epoch_iter, interval, warmup)
        self.final_lrs = self._build_final_lrs(final_lr)
        self.cycle_step = int(cycle_step)
        self.reduce_lr_diff_ratio = float(reduce_lr_diff_ratio)
        self.gaps = [
            base_lr - final_lr
            for base_lr, final_lr in zip(self.base_lrs, self.final_lrs)
        ]
        self._refresh_cycle_shape()

    def _build_final_lrs(self, final_lr):
        """Return lower-bound lr values aligned with optimizer param groups."""
        if isinstance(final_lr, (list, tuple)):
            if len(final_lr) != len(self.base_lrs):
                raise ValueError(
                    "final_lr list length must match optimizer param groups")
            return [float(lr) for lr in final_lr]
        return [float(final_lr) for _ in self.base_lrs]

    def _refresh_cycle_shape(self):
        """Recompute step counts derived from cycle_step and epoch_iter."""
        self.cycle_iter = self.cycle_step * self.epoch_iter
        self.step_size = max(self.cycle_iter // 2, 1)

    def _get_lrs(self, current_step):
        """Return the triangular lr for a global optimizer step."""
        point = current_step % self.cycle_iter
        cycle_index = current_step // self.cycle_iter
        lrs = []
        for base_lr, min_lr, gap in zip(self.base_lrs, self.final_lrs,
                                        self.gaps):
            max_lr = min_lr + gap * self.reduce_lr_diff_ratio**cycle_index
            if point <= self.step_size:
                lr = min_lr + (max_lr - min_lr) * point / self.step_size
            else:
                lr = max_lr - (max_lr - min_lr) * (
                    point - self.step_size) / self.step_size
            lrs.append(lr)
        return lrs

    def _step_step(self, current_step, epoch=None):
        self.set_lrs(self._get_lrs(current_step))
        return SchedulerResult(lr_changed=True)

    def set_runtime_config(self, num_epochs=None, epoch_iter=None):
        super().set_runtime_config(num_epochs=num_epochs,
                                   epoch_iter=epoch_iter)
        if epoch_iter is not None:
            self._refresh_cycle_shape()


class NoOpScheduler(LRSchedulerBase):
    """Scheduler placeholder used when lr should stay unchanged."""

    def __init__(self, optimizer, num_epochs, epoch_iter, **kwargs):
        super().__init__(optimizer, num_epochs, epoch_iter, interval="none")


class MarginScheduler:
    """Update a speaker-classification margin during training.

    This is not an lr scheduler. It is useful for speaker identity auxiliary
    losses that use margin-based classifiers such as AAM/AM softmax. Those
    heads often expose ``projection.update(margin=...)``.
    """

    def __init__(
        self,
        model,
        epoch_iter,
        increase_start_epoch,
        fix_start_epoch,
        initial_margin,
        final_margin,
        update_margin,
        increase_type="exp",
    ):
        """Create a margin schedule.

        Args:
            model: Model with an optional ``projection.update`` method.
            epoch_iter: Number of optimizer steps in one epoch.
            increase_start_epoch: First epoch where the margin starts moving.
            fix_start_epoch: First epoch where the final margin is fixed.
            initial_margin: Margin used before the schedule starts.
            final_margin: Margin used after ``fix_start_epoch``.
            update_margin: If false, ``step`` is a no-op.
            increase_type: ``"exp"`` or any other value for linear growth.
        """
        self.model = model
        # Convert epoch boundaries to iteration boundaries because step() is
        # called with the global batch index.
        self.increase_start_iter = (increase_start_epoch - 1) * epoch_iter
        self.fix_start_iter = (fix_start_epoch - 1) * epoch_iter
        self.initial_margin = initial_margin
        self.final_margin = final_margin
        self.increase_type = increase_type

        self.fix_already = False
        self.current_iter = 0
        self.update_margin = update_margin and hasattr(self.model.projection,
                                                       "update")
        self.increase_iter = self.fix_start_iter - self.increase_start_iter

        self.init_margin()

    def init_margin(self):
        """Set the initial margin if the model supports margin updates."""
        if hasattr(self.model.projection, "update"):
            self.model.projection.update(margin=self.initial_margin)

    def get_increase_margin(self):
        """Return the margin value for the current iteration."""
        initial_val = 1.0
        final_val = 1e-3

        current_iter = self.current_iter - self.increase_start_iter

        if self.increase_type == "exp":  # exponentially increase the margin
            ratio = (1.0 - math.exp(
                (current_iter / self.increase_iter) *
                math.log(final_val / (initial_val + 1e-6))) * initial_val)
        else:  # linearly increase the margin
            ratio = 1.0 * current_iter / self.increase_iter
        return (self.initial_margin +
                (self.final_margin - self.initial_margin) * ratio)

    def step(self, current_iter=None):
        if not self.update_margin or self.fix_already:
            return

        if current_iter is not None:
            self.current_iter = current_iter

        # Before increase_start_iter the initial margin is kept. During the
        # active window the margin moves toward final_margin, then stays fixed.
        if self.current_iter >= self.fix_start_iter:
            self.fix_already = True
            if hasattr(self.model.projection, "update"):
                self.model.projection.update(margin=self.final_margin)
        elif self.current_iter >= self.increase_start_iter:
            if hasattr(self.model.projection, "update"):
                self.model.projection.update(margin=self.get_increase_margin())

        self.current_iter += 1

    def get_margin(self):
        try:
            margin = self.model.projection.margin
        except Exception:
            margin = 0.0

        return margin


BaseClass = LRSchedulerBase

_SCHEDULER_CLASSES = {
    "ExponentialDecrease": ExponentialDecrease,
    "PlateauHalf": PlateauHalf,
    "StepDecay": StepDecay,
    "TriAngular2": TriAngular2,
    "none": NoOpScheduler,
    "NoOpScheduler": NoOpScheduler,
}
assert tuple(_SCHEDULER_CLASSES) == SUPPORTED_SCHEDULERS


def build_scheduler(optimizer,
                    scheduler_conf,
                    scheduler_args=None,
                    num_epochs=None,
                    epoch_iter=None):
    """Build one scheduler from the recipe YAML.

    Args:
        optimizer: Optimizer whose lr will be controlled.
        scheduler_conf: Dict with ``name``, optional ``interval``, optional
            ``warmup``, and scheduler-specific ``args``.
        scheduler_args: Must be empty. Kept only because train.py still passes
            this argument position.
        num_epochs: Total planned training epochs.
        epoch_iter: Number of optimizer steps in one epoch.

    Returns:
        An LRSchedulerBase instance.
    """
    if scheduler_conf is None:
        return NoOpScheduler(optimizer, num_epochs, epoch_iter)

    if scheduler_args:
        raise ValueError(
            "scheduler_args is not supported; put args under scheduler.*.args")
    if not isinstance(scheduler_conf, dict):
        raise TypeError("scheduler config must be a dict or null")

    conf = dict(scheduler_conf)
    name = conf.get("name", "none")
    args = dict(conf.get("args", {}) or {})

    warmup = conf.get("warmup", None)
    interval = conf.get("interval", None)
    if interval is None:
        interval = "validation" if name == "PlateauHalf" else "step"

    if name not in _SCHEDULER_CLASSES:
        raise ValueError("Unsupported scheduler: {}. Supported: {}".format(
            name, ", ".join(SUPPORTED_SCHEDULERS)))
    scheduler_cls = _SCHEDULER_CLASSES[name]
    return scheduler_cls(
        optimizer,
        num_epochs=num_epochs,
        epoch_iter=epoch_iter,
        interval=interval,
        warmup=warmup,
        **args,
    )


def show_lr_curve(scheduler):
    """Plot lr values for a step-based scheduler."""
    import matplotlib.pyplot as plt

    lr_list = []
    for current_step in range(0, scheduler.main_total_steps()):
        scheduler.step_batch(current_step)
        lr_list.append(scheduler.get_lr())
    data_index = list(range(1, len(lr_list) + 1))

    plt.plot(data_index, lr_list, "-o", markersize=1)
    plt.legend(loc="best")
    plt.xlabel("Iteration")
    plt.ylabel("LR")

    plt.show()


if __name__ == "__main__":
    import torch

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ExponentialDecrease(
        optimizer,
        num_epochs=6,
        epoch_iter=500,
        final_lr=0.0001,
        warmup={
            "by": "epoch",
            "epochs": 2,
            "start_factor": 0.0,
        },
    )
    show_lr_curve(scheduler)
