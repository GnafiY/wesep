import os
import re
from typing import List, Optional

import torch

from wesep.utils.schedulers import BaseClass

CHECKPOINT_RE = re.compile(r"^checkpoint_(\d+)\.pt$")


def checkpoint_filename(epoch):
    return "checkpoint_{}.pt".format(epoch)


def get_checkpoint_epoch(filename):
    match = CHECKPOINT_RE.match(os.path.basename(filename))
    return int(match.group(1)) if match else None


def get_checkpoint_link_epoch(model_dir, link_name):
    link_path = os.path.join(model_dir, link_name)
    if not os.path.islink(link_path):
        return None
    return get_checkpoint_epoch(os.readlink(link_path))


class CheckpointManager:

    def __init__(self, model_dir, recent_limit=0, metadata=None, logger=None):
        metadata = metadata or {}
        self.model_dir = model_dir
        self.recent_limit = max(int(recent_limit), 0)
        self.logger = logger
        self.recent_epochs = [
            int(epoch) for epoch in metadata.get("recent_epochs", [])
        ]
        self.periodic_epochs = set(
            int(epoch) for epoch in metadata.get("periodic_epochs", []))
        self.best_epoch = metadata.get("best_epoch", None)
        if self.best_epoch is None:
            self.best_epoch = get_checkpoint_link_epoch(
                self.model_dir, "best_checkpoint.pt")
        self.final_epoch = metadata.get("final_epoch", None)
        if self.final_epoch is None:
            self.final_epoch = get_checkpoint_link_epoch(
                self.model_dir, "final_checkpoint.pt")
        self.latest_epoch = metadata.get("latest_epoch", None)
        if self.latest_epoch is None:
            self.latest_epoch = get_checkpoint_link_epoch(
                self.model_dir, "latest_checkpoint.pt")

    def checkpoint_path(self, epoch):
        return os.path.join(self.model_dir, checkpoint_filename(epoch))

    def _update_link(self, link_name, epoch):
        link_path = os.path.join(self.model_dir, link_name)
        if os.path.lexists(link_path):
            os.remove(link_path)
        os.symlink(checkpoint_filename(epoch), link_path)

    def _protected_epochs(self):
        protected = set(self.recent_epochs)
        protected.update(self.periodic_epochs)
        for epoch in (self.best_epoch, self.final_epoch, self.latest_epoch):
            if epoch is not None:
                protected.add(epoch)
        return protected

    def _delete_if_unprotected(self, epoch):
        if epoch is None or epoch in self._protected_epochs():
            return

        checkpoint_path = self.checkpoint_path(epoch)
        if os.path.isfile(
                checkpoint_path) and not os.path.islink(checkpoint_path):
            os.remove(checkpoint_path)
            if self.logger is not None:
                self.logger.info("Removed stale checkpoint: {}".format(
                    checkpoint_filename(epoch)))

    def prepare_checkpoint(self,
                           epoch,
                           is_best=False,
                           is_periodic=False,
                           is_final=False):
        delete_candidates = []
        old_best_epoch = self.best_epoch
        old_final_epoch = self.final_epoch

        self.latest_epoch = epoch
        if is_best:
            self.best_epoch = epoch
            if old_best_epoch != epoch:
                delete_candidates.append(old_best_epoch)
        if is_final:
            self.final_epoch = epoch
            if old_final_epoch != epoch:
                delete_candidates.append(old_final_epoch)
        if is_periodic:
            self.periodic_epochs.add(epoch)

        if self.recent_limit > 0:
            self.recent_epochs = [
                recent_epoch for recent_epoch in self.recent_epochs
                if recent_epoch != epoch
            ]
            self.recent_epochs.append(epoch)
            while len(self.recent_epochs) > self.recent_limit:
                delete_candidates.append(self.recent_epochs.pop(0))

        return delete_candidates

    def commit_checkpoint(self,
                          epoch,
                          delete_candidates=None,
                          is_best=False,
                          is_final=False):
        self._update_link("latest_checkpoint.pt", epoch)
        if is_best:
            self._update_link("best_checkpoint.pt", epoch)
        if is_final:
            self._update_link("final_checkpoint.pt", epoch)

        for candidate_epoch in delete_candidates or []:
            self._delete_if_unprotected(candidate_epoch)

    def metadata(self, extra=None):
        metadata = {
            "recent_epochs": list(self.recent_epochs),
            "periodic_epochs": sorted(self.periodic_epochs),
            "best_epoch": self.best_epoch,
            "final_epoch": self.final_epoch,
            "latest_epoch": self.latest_epoch,
        }
        if extra:
            metadata.update(extra)
        return metadata


def load_pretrained_model(model: torch.nn.Module,
                          path: str,
                          type: str = "generator"):
    assert type in ["generator", "discriminator"]
    states = torch.load(
        path,
        map_location="cpu",
    )
    if type == "generator":
        state = states["models"][0]
    else:
        assert len(states["models"]) == 2
        state = states["models"][1]

    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state)
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)


def load_checkpoint(
    models: List[torch.nn.Module],
    optimizers: List[torch.optim.Optimizer],
    schedulers: List[BaseClass],
    scaler: Optional[torch.cuda.amp.GradScaler],
    path: str,
    only_model: bool = False,
    mode: str = "all",
):
    assert mode in ["all", "generator", "discriminator"]
    states = torch.load(
        path,
        map_location="cpu",
    )
    if mode == "generator":
        model_state, optimizer_state, scheduler_state = (
            [states["models"][0]],
            [states["optimizers"][0]],
            [states["schedulers"][0]],
        )
    elif mode == "discriminator":
        model_state, optimizer_state, scheduler_state = (
            [states["models"][1]],
            [states["optimizers"][1]],
            [states["schedulers"][1]],
        )
    else:
        model_state, optimizer_state, scheduler_state = (
            states["models"],
            states["optimizers"],
            states["schedulers"],
        )

    for model, state in zip(models, model_state):
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(state, strict=False)
        elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(state, strict=False)
        else:
            model.load_state_dict(state, strict=False)
    if not only_model:
        for optimizer, state in zip(optimizers, optimizer_state):
            optimizer.load_state_dict(state)
        for scheduler, state in zip(schedulers, scheduler_state):
            if scheduler is not None:
                scheduler.load_state_dict(state)
        if scaler is not None:
            if states["scaler"] is not None:
                scaler.load_state_dict(states["scaler"])
    return states.get("metadata", {})


def save_checkpoint(
    models: List[torch.nn.Module],
    optimizers: List[torch.optim.Optimizer],
    schedulers: List[BaseClass],
    scaler: Optional[torch.cuda.amp.GradScaler],
    path: str,
    metadata: Optional[dict] = None,
):
    if isinstance(models[0], torch.nn.DataParallel):
        state_dict = [model.module.state_dict() for model in models]
    elif isinstance(models[0], torch.nn.parallel.DistributedDataParallel):
        state_dict = [model.module.state_dict() for model in models]
    else:
        state_dict = [model.state_dict() for model in models]
    torch.save(
        {
            "models":
            state_dict,
            "optimizers": [o.state_dict() for o in optimizers],
            "schedulers":
            [s.state_dict() if s is not None else None for s in schedulers],
            "scaler":
            scaler.state_dict() if scaler is not None else None,
            "metadata":
            metadata or {},
        },
        path,
    )
