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

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput
from transformers.utils import is_torchdynamo_compiling, logging

from .configuration_eo1 import EO1VisionFlowMatchingConfig
from .modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

logger = logging.get_logger(__name__)


def create_sinusoidal_pos_embedding(
    time: torch.tensor,
    dimension: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
    device="cpu",
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    fraction = torch.linspace(0.0, 1.0, dimension // 2, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)


def sample_noise(shape, device):
    noise = torch.normal(
        mean=0.0,
        std=1.0,
        size=shape,
        dtype=torch.float32,
        device=device,
    )
    return noise


def sample_time(bsize, device):
    time_beta = sample_beta(1.5, 1.0, bsize, device)
    time = time_beta * 0.999 + 0.001
    return time.to(dtype=torch.float32, device=device)


@dataclass
class EO1VisionFlowMatchingOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    fm_loss: torch.FloatTensor | None = None
    ar_loss: torch.FloatTensor | None = None
    actions: torch.FloatTensor | None = None

    logits: torch.FloatTensor | None = None
    past_key_values: list[torch.FloatTensor] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    rope_deltas: torch.LongTensor | None = None
    meta_states: torch.FloatTensor | None = None


class EO1VisionActionProjector(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        activation_layer: str = "linear",
        bias: bool = True,
        device: Any = None,
        dtype: torch.dtype = torch.float32,
    ):
        layers = []
        in_dim = in_channels
        hidden_channels = [in_dim] * (num_layers - 1) + [out_channels]

        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias, dtype=dtype, device=device))
            layers.append(ACT2FN[activation_layer])
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias, dtype=dtype, device=device))
        super().__init__(*layers)

    @property
    def dtype(self):
        return self[0].weight.dtype


class EO1VisionFlowMatchingModel(Qwen2_5_VLForConditionalGeneration):
    config_class = EO1VisionFlowMatchingConfig

    def __init__(
        self,
        config: EO1VisionFlowMatchingConfig,
        build_projector: bool = True,
    ):
        super().__init__(config)

        if build_projector:
            self.build_projector()

    def build_projector(self, dtype=None, device=None):
        device = device or self.device
        dtype = dtype or self.dtype
        hidden_size = self.config.text_config.hidden_size
        max_action_dim = self.config.max_action_dim
        self.state_proj = nn.Linear(max_action_dim, hidden_size).to(dtype=dtype, device=device)
        self.action_in_proj = nn.Linear(max_action_dim, hidden_size).to(dtype=dtype, device=device)
        self.action_out_proj = EO1VisionActionProjector(
            hidden_size,
            max_action_dim,
            self.config.num_action_layers,
            self.config.action_act,
            dtype=dtype,
            device=device,
        )

        self.action_time_mlp_in = nn.Linear(hidden_size * 2, hidden_size).to(dtype=dtype, device=device)
        self.action_time_mlp_out = nn.Linear(hidden_size, hidden_size).to(dtype=dtype, device=device)

    def _has_action_gen_seq(
        self,
        input_ids: torch.LongTensor = None,
    ):
        """Check if the input_ids has action generation sequence."""
        if input_ids is None:
            return False, None
        action_token_id = self.config.text_config.action_token_id
        mask = input_ids == action_token_id
        return mask.any()

    def replace_special_embeddings(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        special_features: torch.FloatTensor = None,
        special_token_ids: torch.LongTensor = None,
    ) -> torch.LongTensor:
        """Replace the special embeddings with the special features."""
        if special_features is not None and special_token_ids is not None:
            n_special_tokens = (input_ids == special_token_ids).sum().item()
            n_special_features = special_features.shape[0]

            assert n_special_tokens == n_special_features, (
                f"Special features and special tokens {special_token_ids} do not match: \
                tokens: {n_special_tokens}, features {n_special_features}"
            )
            mask = input_ids == special_token_ids
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            special_mask = mask_expanded.to(inputs_embeds.device)
            special_features = special_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_mask, special_features)
        return inputs_embeds, None

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        rope_deltas: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
        states: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        action_is_pad: torch.Tensor | None = None,
    ) -> tuple | EO1VisionFlowMatchingOutputWithPast:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                inputs_embeds, _ = self.replace_special_embeddings(
                    input_ids, inputs_embeds, image_embeds, self.config.image_token_id
                )

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                inputs_embeds, _ = self.replace_special_embeddings(
                    input_ids, inputs_embeds, video_embeds, self.config.video_token_id
                )

            if states is not None:
                states = states.type(self.state_proj.weight.dtype)
                state_embs = self.state_proj(states)
                inputs_embeds, _ = self.replace_special_embeddings(
                    input_ids, inputs_embeds, state_embs, self.config.text_config.state_token_id
                )

            if actions is not None:
                noise_mask = input_ids == self.config.text_config.action_token_id
                pass_mask = input_ids == self.config.text_config.action_pass_id
                mask = noise_mask | pass_mask  # (b s)

                pass_mask_in_action = pass_mask[mask]  # (n, )
                pass_mask_in_action = pass_mask_in_action.reshape(*actions.shape[:2], 1)  # (b, h, 1)

                time = sample_time(actions.shape[0], inputs_embeds.device)  # (n,)
                time_expanded = time[:, None, None].repeat(1, actions.shape[1], 1)  # (b, h, 1)
                time_expanded[pass_mask_in_action] = 0.0

                noise = sample_noise(actions.shape, inputs_embeds.device)
                x_t = time_expanded * noise + (1 - time_expanded) * actions
                u_t = noise - actions

                time_embs = create_sinusoidal_pos_embedding(
                    time,
                    self.config.text_config.hidden_size,
                    device=inputs_embeds.device,
                )
                time_embs = time_embs.type(inputs_embeds.dtype)

                x_t = x_t.type(self.action_in_proj.weight.dtype)
                action_embs = self.action_in_proj(x_t)
                time_embs = time_embs[:, None, :].expand_as(action_embs)

                action_time_embs = torch.cat([action_embs, time_embs], dim=2)
                action_time_embs = self.action_time_mlp_in(action_time_embs)
                action_time_embs = F.silu(action_time_embs)
                action_time_embs = self.action_time_mlp_out(action_time_embs)

                num_actions = mask.sum().item()
                num_action_features = action_time_embs.shape[0] * action_time_embs.shape[1]
                assert num_actions == num_action_features, (
                    f"action features and tokens do not match: {num_actions=}, {num_action_features=}"
                )

                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                action_mask = mask_expanded.to(inputs_embeds.device)

                action_time_embs = action_time_embs.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(action_mask, action_time_embs)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        if position_ids is None:
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                else:
                    delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                position_ids += delta.to(position_ids.device)

        model_kwargs = {
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "use_cache": False if self.training else use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
            "cache_position": cache_position,
        }

        _actions = None
        if not (self.training or states is None) and actions is None and self._has_action_gen_seq(input_ids):
            _actions, outputs = self._sample_actions(input_ids=input_ids, **model_kwargs)
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states[:, -1])
        else:
            outputs = self.model(**model_kwargs)
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)

        loss = None
        fm_loss = None
        v_t = None
        if actions is not None:
            action_time_embs = hidden_states[action_mask[..., 0]]
            action_time_embs = action_time_embs.type(self.action_out_proj.dtype)
            v_t = self.action_out_proj(action_time_embs)
            u_t = u_t.reshape(v_t.shape)
            v_t = v_t.type(u_t.dtype)
            losses = F.mse_loss(u_t, v_t, reduction="none")
            if action_is_pad is not None:
                in_episode_bound = (~action_is_pad).reshape(-1, 1)
                losses = losses * in_episode_bound

            in_denoise_bound = (~pass_mask_in_action).reshape(-1, 1)
            losses = losses * in_denoise_bound
            fm_loss = losses.mean()
            loss = fm_loss

        ar_loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            ar_loss = loss_fct(shift_logits, shift_labels)
            loss = loss + ar_loss if loss is not None else ar_loss

        return EO1VisionFlowMatchingOutputWithPast(
            loss=loss,
            fm_loss=fm_loss,
            ar_loss=ar_loss,
            actions=_actions,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    @torch.no_grad()
    def _sample_actions(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> Tensor:
        """Sample actions from the model, break down into 2 steps to make a unified generation interface:
        1. pass the mm prefix to the model, and update kvcache
        2. perform denoising steps, with noise q and mm kvcache
        input_ids:
        <|im_start|>user<|vision_start|><|image_pad|>...<|vision_end|><|state_start|><|state_pad|><|state_end|>task...<|vla|><|im_end|> -> AR kvcache
        <|im_start|>assistant<|action_start|><|action_pad|>...<|action_end|> -> FM denoising
        <|im_end|> -> AR
        """
        chunksz_eoa = self.config.action_chunk_size + 1
        mm_outputs = self.model(
            position_ids=position_ids[..., :-chunksz_eoa],
            attention_mask=attention_mask[:, :-chunksz_eoa],
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds[:, :-chunksz_eoa],
            use_cache=use_cache,
            cache_position=cache_position[:-chunksz_eoa],
        )
        device = inputs_embeds.device
        x_t = sample_noise(
            [
                inputs_embeds.shape[0],
                self.config.action_chunk_size,
                self.config.max_action_dim,
            ],
            device,
        )
        x_t = x_t.type(self.action_in_proj.weight.dtype)

        dt = torch.tensor(-1.0 / self.config.num_denoise_steps, device=device)
        time = torch.ones(inputs_embeds.shape[0], device=device)
        pass_seq_length = past_key_values.get_seq_length()

        action_mask = input_ids == self.config.text_config.action_token_id
        while time >= -dt / 2:
            time_embs = create_sinusoidal_pos_embedding(
                time,
                self.config.text_config.hidden_size,
                device=device,
            )
            time_embs = time_embs.type(inputs_embeds.dtype)
            action_embs = self.action_in_proj(x_t)
            time_embs = time_embs[:, None, :].expand_as(action_embs)

            action_time_embs = torch.cat([action_embs, time_embs], dim=2)
            action_time_embs = self.action_time_mlp_in(action_time_embs)
            action_time_embs = F.silu(action_time_embs)
            action_time_embs = self.action_time_mlp_out(action_time_embs)
            action_time_embs = action_time_embs.to(device, inputs_embeds.dtype)
            inputs_embeds[action_mask] = action_time_embs

            past_key_values.crop(pass_seq_length)
            outputs = self.model(
                position_ids=position_ids[..., -chunksz_eoa:],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds[:, -chunksz_eoa:],
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position[-chunksz_eoa:],
            )

            hidden_states = outputs[0]
            action_time_embs = hidden_states[:, :-1]  # exclude <eoa>
            action_time_embs = action_time_embs.type(self.action_out_proj.dtype)
            v_t = self.action_out_proj(action_time_embs)

            # euler step
            x_t += dt * v_t.reshape(x_t.shape)
            time += dt
        outputs.last_hidden_state = torch.cat(
            [mm_outputs.last_hidden_state, outputs.last_hidden_state], dim=1
        )
        return (x_t, outputs)


EO1VisionFlowMatchingModel.register_for_auto_class()
