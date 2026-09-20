# Copyright (c) 2023 Shuai Wang (wsstriving@gmail.com)
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

import logging
import os
from pprint import pformat

import fire
import matplotlib.pyplot as plt
import tableprint as tp
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader

import wesep.utils.schedulers as schedulers
from wesep.dataset.dataset import Dataset
from wesep.dataset.collate import (
    BASE_COLLECT_KEYS,
    build_collect_keys,
    tse_collate_fn,
)
from wesep.models import get_model
from wesep.modules.common.deep_update import deep_update
from wesep.utils.checkpoint import (
    CheckpointManager,
    checkpoint_filename,
    get_checkpoint_epoch,
    load_checkpoint,
    load_pretrained_model,
    save_checkpoint,
)
from wesep.utils.executor import Executor
from wesep.utils.losses import LossManager
from wesep.utils.torch_compat import create_cuda_grad_scaler
from wesep.utils.utils import parse_config_or_kwargs, set_seed, setup_logger
from wesep.utils.file_utils import load_json, load_yaml

MAX_NUM_log_files = 100  # The maximum number of log-files to be kept
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def train(config="conf/config.yaml", **kwargs):
    """Trains a model on the given features and spk labels.

    :config: A training configuration. Note that all parameters in the
             config can also be manually adjusted with --ARG VALUE
    :returns: None
    """
    configs = parse_config_or_kwargs(config, **kwargs)
    labels = load_yaml(configs["train_cues"]).get("labels", {})
    if "speaker_label" in labels:
        spk2label = load_json(labels["speaker_label"]["resource"])
        configs["model_args"]["tse_model"]["num_speakers"] = len(spk2label)
    checkpoint = configs.get("checkpoint", None)
    if checkpoint is not None:
        checkpoint = os.path.realpath(checkpoint)
    find_unused_parameters = configs.get("find_unused_parameters", False)

    # Initialize distributed training before constructing CUDA modules.
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    gpu = int(configs["gpus"][rank])
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend="nccl")

    # Prepare rank-aware logging and the model output directory.
    model_dir = os.path.join(configs["exp_dir"], "models")
    logger = setup_logger(rank, configs["exp_dir"], gpu, MAX_NUM_log_files)

    if world_size > 1:
        logger.info("training on multiple gpus, this gpu {}".format(gpu))

    if rank == 0:
        logger.info("exp_dir is: {}".format(configs["exp_dir"]))
        logger.info("<== Passed Arguments ==>")
        # Log resolved arguments for reproducibility.
        for line in pformat(configs).split("\n"):
            logger.info(line)

    # Make randomness rank-dependent but reproducible.
    set_seed(configs["seed"] + rank)

    # Build train/validation datasets and infer epoch lengths.
    train_dataset_args = configs["dataset_args"]
    val_dataset_args = deep_update(
        train_dataset_args,
        configs.get("val_dataset_args", {}),
        inplace=False,
    )
    train_dataset = Dataset(
        configs["data_type"],
        configs["train_data"],
        train_dataset_args,
        state="train",
        repeat_dataset=configs.get("repeat_dataset", True),
        cues_yaml=configs.get("train_cues", None),
    )
    val_dataset = Dataset(
        configs["data_type"],
        configs["val_data"],
        val_dataset_args,
        state="val",
        repeat_dataset=True,
        cues_yaml=configs.get("val_cues", None),
    )
    train_collect_keys = build_collect_keys(
        load_yaml(configs["train_cues"]),
        train_dataset_args,
        BASE_COLLECT_KEYS,
    )
    val_collect_keys = build_collect_keys(
        load_yaml(configs["val_cues"]),
        val_dataset_args,
        BASE_COLLECT_KEYS,
    )
    train_dataloader = DataLoader(
        train_dataset,
        **configs["dataloader_args"],
        collate_fn=lambda batch: tse_collate_fn(batch, train_collect_keys),
    )
    val_dataloader = DataLoader(
        val_dataset,
        **configs["dataloader_args"],
        collate_fn=lambda batch: tse_collate_fn(batch, val_collect_keys),
    )
    batch_size = configs["dataloader_args"]["batch_size"]
    if configs["dataset_args"].get("sample_num_per_epoch", 0) > 0:
        sample_num_per_epoch = configs["dataset_args"]["sample_num_per_epoch"]
    else:
        with open(configs["train_samples"], "r", encoding="utf-8") as f:
            sample_num_per_epoch = sum(1 for _ in f)
    epoch_iter = sample_num_per_epoch // world_size // batch_size
    with open(configs["val_samples"], "r", encoding="utf-8") as f:
        val_sample_num = sum(1 for _ in f)
    val_iter = val_sample_num // world_size // batch_size

    if rank == 0:
        logger.info("<== Dataloaders ==>")
        logger.info("train dataloaders created")
        logger.info("epoch iteration number: {}".format(epoch_iter))
        logger.info("val iteration number: {}".format(val_iter))

    # Keep list containers for checkpoint and executor compatibility.
    model_list = []
    scheduler_list = []
    optimizer_list = []

    # Build the model and wrap it with DDP before optimizer creation.
    logger.info("<== Model ==>")
    model_init = configs.get("model_init", {}).get("tse_model", None)
    defer_pretrained = checkpoint is not None or model_init is not None
    model = get_model(configs["model"]["tse_model"])(
        configs["model_args"]["tse_model"],
        defer_pretrained=defer_pretrained,
    )
    num_params = sum(param.numel() for param in model.parameters())

    if rank == 0:
        logger.info("tse_model size: {:.2f} M".format(num_params / 1e6))
        for line in pformat(model).split("\n"):
            logger.info(line)

    model.cuda()
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, find_unused_parameters=find_unused_parameters)
    device = torch.device("cuda")

    # Build named train/validation losses after the device is selected.
    loss_conf = configs.get("loss") or [{
        "type": "SISDR",
        "output": "speech",
        "target": "wav_target",
        "weight": 1.0,
        "stages": ["train", "val"],
    }]
    loss_manager = LossManager(loss_conf).to(device)

    if rank == 0:
        logger.info("<== TSE Model Loss ==>")
        logger.info("loss configuration is: " + str(loss_conf))

    optimizer = getattr(torch.optim, configs["optimizer"]["tse_model"])(
        ddp_model.parameters(), **configs["optimizer_args"]["tse_model"])
    if rank == 0:
        logger.info("<== TSE Model Optimizer ==>")
        logger.info("optimizer is: " + configs["optimizer"]["tse_model"])

    # Build the lr scheduler after optimizer so it can read the base lr.
    scheduler_root = configs.get("scheduler") or {}
    scheduler_args_root = configs.get("scheduler_args") or {}
    if isinstance(scheduler_root, dict):
        scheduler_conf = scheduler_root.get("tse_model", None)
    else:
        scheduler_conf = scheduler_root
    if isinstance(scheduler_args_root, dict):
        scheduler_args = scheduler_args_root.get("tse_model", None)
    else:
        scheduler_args = scheduler_args_root
    scheduler = schedulers.build_scheduler(
        optimizer,
        scheduler_conf,
        scheduler_args,
        num_epochs=configs["num_epochs"],
        epoch_iter=epoch_iter,
    )
    if rank == 0:
        logger.info("<== TSE Model Scheduler ==>")
        logger.info("scheduler is: {}".format(scheduler))

    if model_init is not None:
        logger.info("Load initial model from {}".format(model_init))
        load_pretrained_model(ddp_model, model_init)
    elif checkpoint is None:
        logger.info("Train model from scratch ...")

    model_list.append(ddp_model)
    optimizer_list.append(optimizer)
    scheduler_list.append(scheduler)

    scaler = create_cuda_grad_scaler(configs["enable_amp"])

    # Restore training state and checkpoint metadata when resuming.
    checkpoint_metadata = {}
    if checkpoint is not None:
        checkpoint_metadata = load_checkpoint(model_list, optimizer_list,
                                              scheduler_list, scaler,
                                              checkpoint) or {}
        checkpoint_epoch = get_checkpoint_epoch(checkpoint)
        if checkpoint_epoch is None:
            raise ValueError(
                "Invalid checkpoint filename: {}".format(checkpoint))
        start_epoch = checkpoint_epoch + 1
        logger.info("Load checkpoint: {}".format(checkpoint))
    else:
        start_epoch = 1
    logger.info("start_epoch: {}".format(start_epoch))

    # Store the resolved config next to training logs.
    if rank == 0:
        saved_config_path = os.path.join(configs["exp_dir"], "config.yaml")
        with open(saved_config_path, "w") as fout:
            data = yaml.dump(configs)
            fout.write(data)

    # Main train/validate/checkpoint loop.
    dist.barrier(device_ids=[gpu])  # wait for rank 0 log setup
    if rank == 0:
        logger.info("<========== Training process ==========>")
        header = ["Train/Val", "Epoch", "iter", "Loss", "LR"]
        for line in tp.header(header, width=10, style="grid").split("\n"):
            logger.info(line)
    dist.barrier(device_ids=[gpu])  # align ranks before first epoch

    executor = Executor()
    executor.step = 0

    train_losses = []
    val_losses = []
    best_loss = checkpoint_metadata.get("best_loss", None)
    num_recent_checkpoints = max(int(configs.get("num_avg", 0)), 0)
    checkpoint_manager = None
    if rank == 0:
        checkpoint_manager = CheckpointManager(
            model_dir,
            recent_limit=num_recent_checkpoints,
            metadata=checkpoint_metadata,
            logger=logger,
        )
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, configs["num_epochs"] + 1):
        last_epoch = epoch
        train_dataset.set_epoch(epoch)

        # Train one epoch, then evaluate the validation split.
        train_loss, _ = executor.train(
            train_dataloader,
            model_list,
            epoch_iter,
            optimizer_list,
            loss_manager,
            scheduler_list,
            scaler=scaler,
            epoch=epoch,
            logger=logger,
            enable_amp=configs["enable_amp"],
            clip_grad=configs["clip_grad"],
            log_batch_interval=configs["log_batch_interval"],
            device=device,
        )

        val_loss, _ = executor.cv(
            val_dataloader,
            model_list,
            val_iter,
            loss_manager,
            epoch=epoch,
            logger=logger,
            enable_amp=configs["enable_amp"],
            log_batch_interval=configs["log_batch_interval"],
            device=device,
        )

        # Scheduler decisions must use identical metrics on every rank.
        if world_size > 1:
            loss_tensor = torch.tensor(
                [float(train_loss), float(val_loss)],
                device=device,
            )
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_tensor /= world_size
            train_loss = loss_tensor[0].item()
            val_loss = loss_tensor[1].item()
        global_step = epoch * epoch_iter

        # Validation/epoch schedulers may update lr or request early stop.
        scheduler_results = []
        for scheduler in scheduler_list:
            if scheduler is not None:
                scheduler_results.append(
                    scheduler.step_validation(val_loss, epoch, global_step))
                scheduler_results.append(
                    scheduler.step_epoch(epoch, global_step))

        is_best = best_loss is None or val_loss < best_loss
        if is_best:
            best_loss = val_loss

        should_stop = any(result.should_stop for result in scheduler_results)
        stop_tensor = torch.tensor(
            int(should_stop),
            device=device,
            dtype=torch.int,
        )
        dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
        should_stop = bool(stop_tensor.item())

        if rank == 0:
            # Track scalar history and update the loss curve.
            logger.info("Epoch {} Train info train_loss {}".format(
                epoch, train_loss))
            logger.info("Epoch {} Val info val_loss {}".format(
                epoch, val_loss))
            for result in scheduler_results:
                if result.message:
                    logger.info(result.message)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Update the train/validation loss plot.
            plt.figure()
            plt.title("Loss of Train and Validation")
            x = list(range(start_epoch, epoch + 1))
            plt.plot(x, train_losses, "b-", label="Train Loss", linewidth=0.8)
            plt.plot(x,
                     val_losses,
                     "c-",
                     label="Validation Loss",
                     linewidth=0.8)
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.xticks(range(start_epoch, epoch + 1, 1))
            plt.savefig(
                f"{configs['exp_dir']}/{configs['model']['tse_model']}.png")
            plt.close()

        if rank == 0:
            # Save this epoch, then let the manager rotate stale checkpoints.
            save_interval = int(configs.get("save_epoch_interval", 1))
            is_periodic = save_interval > 0 and epoch % save_interval == 0
            is_final = should_stop or epoch == configs["num_epochs"]
            should_save = (num_recent_checkpoints > 0 or is_best or is_final
                           or is_periodic)
            if should_save:
                checkpoint_name = checkpoint_filename(epoch)
                delete_candidates = checkpoint_manager.prepare_checkpoint(
                    epoch,
                    is_best=is_best,
                    is_periodic=is_periodic,
                    is_final=is_final,
                )
                metadata = checkpoint_manager.metadata({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_loss": best_loss,
                })
                save_checkpoint(
                    model_list,
                    optimizer_list,
                    scheduler_list,
                    scaler,
                    os.path.join(model_dir, checkpoint_name),
                    metadata=metadata,
                )
                checkpoint_manager.commit_checkpoint(
                    epoch,
                    delete_candidates,
                    is_best=is_best,
                    is_final=is_final,
                )

        if should_stop:
            if rank == 0:
                logger.info("Early stopping at epoch {}".format(epoch))
            break

    if rank == 0:
        if checkpoint_manager.final_epoch != last_epoch:
            delete_candidates = checkpoint_manager.prepare_checkpoint(
                last_epoch, is_final=True)
            checkpoint_manager.commit_checkpoint(
                last_epoch,
                delete_candidates,
                is_final=True,
            )
        logger.info(tp.bottom(len(header), width=10, style="grid"))


if __name__ == "__main__":
    fire.Fire(train)
