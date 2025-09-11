# Copyright 2025 EO-Robotics Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
from transformers import Trainer
from transformers.trainer import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    is_sagemaker_mp_enabled,
    logger,
)
from transformers.trainer_utils import SaveStrategy


class MetaLossesTrainerState(TrainerCallback):
    def __init__(self, meta_losses: list[str]):
        self.meta_losses = meta_losses

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.meta_losses = {k: torch.tensor(0.0).to(args.device) for k in self.meta_losses}
        return control


class OneVisionTrainer(Trainer):
    """Custom Trainer for EOneVision model.
    This class extends the Trainer class from the transformers library to provide
    additional functionality specific to the EOneVision model.
    It includes methods for creating an optimizer with different learning rates for
    different parts of the model, and for handling mixed precision training.
    Args:
        processor (Processor): The processor to use for data processing.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        meta_losses = ["ar_loss", "fm_loss"]
        self.add_callback(MetaLossesTrainerState(meta_losses))

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled() or (self.args.vision_lr is None and self.args.merger_lr is None):
            return super().create_optimizer()

        opt_model = self.model
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            visual_parameters, merger_parameters = [], []

            if self.args.vision_lr is not None:
                visual_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if "visual" in name and "merger" not in name
                ]
            if self.args.merger_lr is not None:
                merger_parameters = [name for name, _ in opt_model.named_parameters() if "merger" in name]

            special_lr_parameters = merger_parameters + visual_parameters

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            if visual_parameters:
                optimizer_grouped_parameters.extend(
                    [
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (n in decay_parameters and n in visual_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.vision_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (n not in decay_parameters and n in visual_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.vision_lr,
                        },
                    ]
                )

            if merger_parameters:
                optimizer_grouped_parameters.extend(
                    [
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (n in decay_parameters and n in merger_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.merger_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (n not in decay_parameters and n in merger_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.merger_lr,
                        },
                    ]
                )

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2**20}M params")

        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if hasattr(self.control, "meta_losses") and model.training:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

            if not isinstance(outputs, dict):
                raise ValueError(
                    "The model output should be a dictionary or ModelOutput and not a tuple or list."
                )

            for k, v in outputs.items():
                if k in self.control.meta_losses and v is not None:  # and self.args.n_gpu > 1:
                    self.control.meta_losses[k] += v.detach().mean() / self.args.gradient_accumulation_steps

            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            # if is_torch_xla_available():
            #     xm.mark_step()
            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            if hasattr(self.control, "meta_losses"):
                for k, v in self.control.meta_losses.items():
                    logs[k] = self._nested_gather(v).mean().item()
                    self.control.meta_losses[k] -= self.control.meta_losses[k]
                    logs[k] = round(logs[k] / (self.state.global_step - self._globalstep_last_logged), 4)

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
