# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright 2025 the LLAMA4, Meta Inc., vLLM, and HuggingFace Inc. team.
# All rights reserved.
#
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
"""Inference-only Dense LLaMA 4 model - a dense version of LLaMA 4 architecture."""
from collections.abc import Iterable
from typing import Any, Optional

import torch
from torch import nn
from transformers import Llama4TextConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .llama import LlamaForCausalLM, LlamaMLP, LlamaModel
from .llama4 import Llama4Attention
from .utils import (AutoWeightsLoader, extract_layer_index,
                    is_pp_missing_parameter)


class DenseLlama4DecoderLayer(nn.Module):
    """
    Decoder layer for Dense LLaMA 4: uses Llama4Attention and LlamaMLP (dense MLP).
    This is the only difference from the original LLaMA 4 (which uses MoE).
    """

    def __init__(
        self,
        config: Llama4TextConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[torch.nn.Module] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.layer_idx = extract_layer_index(prefix)
        self.hidden_size = config.hidden_size
        rope_theta = config.rope_theta
        rope_scaling = config.rope_scaling
        max_position_embeddings = config.max_position_embeddings

        self.self_attn = Llama4Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=False,
            bias_o_proj=False,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )

        # Always use dense MLP instead of the conditional MoE/MLP logic
        self.feed_forward = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size_mlp,
            hidden_act="silu",
            quant_config=quant_config,
            bias=False,
            prefix=f"{prefix}.feed_forward",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(positions=positions,
                                       hidden_states=hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual


@support_torch_compile
class DenseLlama4Model(LlamaModel):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 layer_type: type[DenseLlama4DecoderLayer] = DenseLlama4DecoderLayer):
        super().__init__(vllm_config=vllm_config,
                         prefix=prefix,
                         layer_type=layer_type)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            # Skip expert weights since we're using dense MLP
            if ("experts." in name or "router." in name or
                "shared_expert." in name):
                continue

            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
        return loaded_params


class DenseLlama4ForCausalLM(LlamaForCausalLM):

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # update temperature tuning config from generation config
        hf_config = vllm_config.model_config.hf_config
        if hasattr(hf_config, "generation_config"):
            gen_config = hf_config.generation_config
            if (hasattr(gen_config, "attn_temperature_tuning") and
                gen_config.attn_temperature_tuning is not None):
                hf_config.attn_temperature_tuning = gen_config.attn_temperature_tuning

        super().__init__(vllm_config=vllm_config,
                         prefix=prefix,
                         layer_type=DenseLlama4DecoderLayer)

    def _init_model(self,
                    vllm_config: VllmConfig,
                    prefix: str = "",
                    layer_type: type[DenseLlama4DecoderLayer] = DenseLlama4DecoderLayer):
        return DenseLlama4Model(vllm_config=vllm_config,
                                prefix=prefix,
                                layer_type=layer_type)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)