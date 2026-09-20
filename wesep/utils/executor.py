# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
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

from contextlib import nullcontext

import tableprint as tp
import torch

from wesep.utils.funcs import clip_gradients
from wesep.utils.torch_compat import cuda_autocast


class Executor:

    def __init__(self):
        self.step = 0

    # -------------------------
    # train
    # -------------------------

    def train(self,
              dataloader,
              models,
              epoch_iter,
              optimizers,
              loss_manager,
              schedulers,
              scaler,
              epoch,
              enable_amp,
              logger,
              clip_grad=5.0,
              log_batch_interval=100,
              device=torch.device("cuda")):

        model = models[0]
        optimizer = optimizers[0]

        model.train()
        log_interval = log_batch_interval
        losses = []

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext

        with model_context():
            for i, batch in enumerate(dataloader):

                cur_iter = (epoch - 1) * epoch_iter + i
                for scheduler in schedulers:
                    if scheduler is not None:
                        scheduler.step_batch(cur_iter, epoch=epoch)

                # Move tensor fields without changing bool, integer, or float dtype.
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device)

                with cuda_autocast(enabled=enable_amp):
                    outputs = model(batch)
                    loss = loss_manager(outputs, batch, stage="train")

                losses.append(loss.item())
                total_loss_avg = sum(losses) / len(losses)

                # ---- backward ----
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_gradients(model, clip_grad)
                scaler.step(optimizer)
                scaler.update()

                if (i + 1) % log_interval == 0:
                    logger.info(
                        tp.row(
                            (
                                "TRAIN",
                                epoch,
                                i + 1,
                                total_loss_avg,
                                optimizer.param_groups[0]["lr"],
                            ),
                            width=10,
                            style="grid",
                        ))

                if (i + 1) == epoch_iter:
                    break

        total_loss_avg = sum(losses) / len(losses)
        return total_loss_avg, 0

    # -------------------------
    # cv / validation
    # -------------------------

    def cv(self,
           dataloader,
           models,
           val_iter,
           loss_manager,
           epoch,
           enable_amp,
           logger,
           log_batch_interval=100,
           device=torch.device("cuda")):

        model = models[0]
        model.eval()

        log_interval = log_batch_interval
        losses = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):

                # Move tensor fields without changing bool, integer, or float dtype.
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device)

                with cuda_autocast(enabled=enable_amp):
                    outputs = model(batch)
                    loss = loss_manager(outputs, batch, stage="val")

                losses.append(loss.item())
                total_loss_avg = sum(losses) / len(losses)

                if (i + 1) % log_interval == 0:
                    logger.info(
                        tp.row(
                            ("VAL", epoch, i + 1, total_loss_avg, "-"),
                            width=10,
                            style="grid",
                        ))

                if (i + 1) == val_iter:
                    break

        total_loss_avg = sum(losses) / len(losses)
        return total_loss_avg, 0
