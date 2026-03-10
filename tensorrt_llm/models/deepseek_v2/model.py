# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections import OrderedDict
from typing import Optional, Union

import numpy as np
import tensorrt as trt
import torch
import transformers

from ..._common import default_net
from ..._utils import pad_vocab_size, torch_dtype_to_str
from ...functional import (Tensor, cast, concat, gather_last_token_logits,
                           index_select, non_gated_version, recv, send, shape)
from ...layers import (MOE, AttentionMaskType, ColumnLinear,
                       DeepseekV2Attention, Embedding, GatedMLP, MoeConfig,
                       PositionEmbeddingType, RmsNorm, SharedMoE,
                       SpecDecodingParams)
from ...llmapi.kv_cache_type import KVCacheType
from ...mapping import Mapping
from ...module import Module, ModuleList
from ...plugin import init_all_reduce_helper
from ..model_weights_loader import ModelWeightsLoader
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              PretrainedConfig)
from .config import DeepSeekV2Config, DeepseekV2MTPConfig
from .convert import convert_deepseekv2, load_weights_from_hf_safetensors


class DeepseekV2DecoderLayer(Module):

    def __init__(self, config: PretrainedConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        # Input layernorm in Deepseek v2 is same as Llama
        self.input_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                       eps=config.norm_epsilon,
                                       dtype=config.dtype)

        layers_range = config.mapping.pp_layers(config.num_hidden_layers)
        local_layer_idx = layer_idx - layers_range[0]

        self.attention = DeepseekV2Attention(
            local_layer_idx=local_layer_idx,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            eps=config.norm_epsilon,
            attention_mask_type=AttentionMaskType.causal,
            dtype=config.dtype,
            position_embedding_type=PositionEmbeddingType.learned_absolute,
            rotary_embedding_base=config.rotary_base,
            rotary_embedding_scaling=None,
            rotary_embedding_beta_fast=config.rotary_scaling['beta_fast'],
            rotary_embedding_beta_slow=config.rotary_scaling['beta_slow'],
            rotary_embedding_mscale=config.rotary_scaling['mscale'],
            rotary_embedding_mscale_all_dim=config.
            rotary_scaling['mscale_all_dim'],
            rotary_embedding_origin_max_position=config.
            rotary_scaling['original_max_position_embeddings'],
            rotary_scaling=config.rotary_scaling,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            tp_rank=config.mapping.tp_rank)

        # Added deepseek MoE and shared_experts
        # First decoder layer: MLA + dense MLP + input_layernorm(RMSNorm) + post_attention_layernorm(RMSNorm)
        # Rest decoder layer: MLA + MoE MLP + MoE Gate + shared_experts(MLP) + input_layernorm(RMSNorm) + post_attention_layernorm(RMSNorm)
        # Added MLA in co-testing phase, use standard attention for MoE testing

        # Distinguish dense MLP and MoE MLP
        # dense_config = DenseConfig(intermediate_size=config.intermediate_size)
        moe_config = config.moe
        # In case of moe_config is a dict
        if isinstance(moe_config, dict):
            moe_config = MoeConfig.from_dict(moe_config)

        if moe_config.num_experts > 0 and layer_idx > 0:
            hidden_act = config.hidden_act
            mlp_hidden_size = config.moe_inter_size
            mlp_kwargs = {'moe_config': moe_config, 'mapping': config.mapping}
            if moe_config.shared_expert_intermediate_size > 0:
                ClsMLP = SharedMoE
                mlp_kwargs['use_shared_gate'] = False
                mlp_kwargs['use_side_stream'] = False
            else:
                ClsMLP = MOE
        else:
            ClsMLP = GatedMLP
            mlp_hidden_size = config.intermediate_size
            hidden_act = non_gated_version(
                config.hidden_act)  # back to non gated for dense layers
            mlp_kwargs = {}

        self.mlp = ClsMLP(hidden_size=config.hidden_size,
                          ffn_hidden_size=mlp_hidden_size,
                          hidden_act=hidden_act,
                          dtype=config.dtype,
                          bias=False,
                          tp_group=config.mapping.tp_group,
                          tp_size=config.mapping.tp_size,
                          quant_mode=config.quant_mode,
                          **mlp_kwargs)

        # Pose layernorm in Deepseek v2 is same as Llama
        self.post_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                      eps=config.norm_epsilon,
                                      dtype=config.dtype)

    def forward(self,
                hidden_states,
                attention_mask=None,
                use_cache=False,
                spec_decoding_params=None,
                kv_cache_params=None,
                attention_params=None):

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attention_output = self.attention(
            hidden_states=hidden_states,
            use_cache=use_cache,
            spec_decoding_params=spec_decoding_params,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params)
        if use_cache:
            attention_output, presents = attention_output

        hidden_states = residual + attention_output

        residual_attn = hidden_states

        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual_attn + hidden_states
        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class DeepseekV2Model(Module):

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        init_all_reduce_helper()  # enable use_customer_all_reduce
        self.dtype = config.dtype
        self.mapping = config.mapping
        if self.mapping.is_first_pp_rank():
            self.vocab_embedding = Embedding(config.vocab_size,
                                             config.hidden_size,
                                             dtype=config.dtype)
        self.layers = DecoderLayerList(DeepseekV2DecoderLayer, config)

        if self.mapping.is_last_pp_rank():
            self.ln_f = RmsNorm(normalized_shape=config.hidden_size,
                                eps=config.norm_epsilon,
                                dtype=config.dtype)

        self.head_num = config.num_attention_heads
        self.head_size = config.qk_nope_head_dim + config.qk_rope_head_dim

    def forward(self,
                input_ids,
                position_ids=None,
                use_cache=False,
                attention_mask=None,
                spec_decoding_params=None,
                kv_cache_params=None,
                attention_params=None,
                hidden_states=None,
                prompt_embedding_table: Optional[Tensor] = None,
                prompt_tasks: Optional[Tensor] = None,
                prompt_vocab_size: Optional[Tensor] = None):

        ptuning_args = [
            prompt_embedding_table, prompt_tasks, prompt_vocab_size
        ] if prompt_embedding_table is not None else []

        if self.mapping.is_first_pp_rank():
            hidden_states = self.vocab_embedding(input_ids, *ptuning_args)
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())

        hidden_states = self.layers.forward(
            hidden_states,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            spec_decoding_params=spec_decoding_params)

        if use_cache:
            hidden_states, presents = hidden_states

        if self.mapping.is_last_pp_rank():
            hidden_states = self.ln_f(hidden_states)
        else:
            hidden_states = send(hidden_states, self.mapping.next_pp_rank())

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class DeepseekV2ForCausalLM(DecoderModelForCausalLM):
    config_class = DeepSeekV2Config

    def __init__(self, config: PretrainedConfig):
        transformer = DeepseekV2Model(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size,
                                           config.mapping.tp_size)
        if config.mapping.is_last_pp_rank():
            lm_head = ColumnLinear(config.hidden_size,
                                   vocab_size_padded,
                                   bias=False,
                                   dtype=config.dtype,
                                   tp_group=config.mapping.tp_group,
                                   tp_size=config.mapping.tp_size,
                                   gather_output=True)
        else:
            lm_head = None
        self.mapping = config.mapping
        super().__init__(config, transformer, lm_head)

    @classmethod
    def from_hugging_face(
            cls,
            model_dir,
            dtype: str = 'auto',
            hf_model: Optional[transformers.PreTrainedModel] = None,
            use_preloading: bool = False,
            use_safetensors_loading: bool = False,
            mapping: Optional[Mapping] = None,
            override_fields={},
            **kwargs):

        if mapping is None:
            mapping = Mapping()
        pretrained_config = DeepSeekV2Config.from_hugging_face(model_dir,
                                                               dtype=dtype,
                                                               mapping=mapping,
                                                               **kwargs)
        if dtype == 'auto':
            dtype = getattr(pretrained_config, 'torch_dtype', None)
        if dtype is None:
            dtype = 'float16'
        if isinstance(dtype, torch.dtype):
            dtype = torch_dtype_to_str(dtype)
        if dtype == 'float32':  # should remove "float32"
            dtype = 'float16'
        if dtype == 'bfloat16' and torch.cuda.get_device_properties(
                0).major < 8:
            logger.warning(
                "Pre SM 80 GPUs do not support bfloat16, fallback to float16")
            dtype = 'float16'

        deepseek = cls.from_config(pretrained_config)

        # If use_preloading is True, load the model from hf_model
        # If use_safetensors_loading is True, load the model from safetensors
        # if TRTLLM_DISABLE_UNIFIED_CONVERTER is not set, load the model use unified converter (recommended and default)
        if use_preloading:
            weights = convert_deepseekv2(
                hf_model,
                pretrained_config,
                mapping,
                dtype=dtype,
                use_parallel_embedding=pretrained_config.use_parallel_embedding,
                sharding_dim=pretrained_config.embedding_sharding_dim)
            deepseek.load(weights)
            return deepseek

        if os.environ.get("TRTLLM_DISABLE_UNIFIED_CONVERTER") is None:

            custom_dict = {}
            rank_experts = mapping.ep_experts(pretrained_config.moe.num_experts)
            for index, module in enumerate(deepseek.transformer.layers):

                if pretrained_config.q_lora_rank is not None:
                    module.attention.tllm_to_externel_key_dict = {
                        "fused_q_proj": ["q_b_proj.weight", "kv_b_proj.weight"],
                        "q_b_proj": "q_b_proj.weight",  #v2
                        "q_a_proj": "q_a_proj.weight",  #v2
                        "kv_b_proj": "kv_b_proj.weight",
                        "q_a_layernorm": "q_a_layernorm"
                    }
                    module.attention.fused_a.tllm_to_externel_key_dict = {
                        "fused_a": ["q_a_proj", "kv_a_proj_with_mqa"]
                    }  #v2
                else:
                    module.attention.tllm_to_externel_key_dict = {
                        "fused_q_proj": ["q_proj.weight",
                                         "kv_b_proj.weight"],  #v2 lite
                        "q_b_proj": "q_proj.weight",  #v2 lite
                        "kv_b_proj": "kv_b_proj.weight",
                        "q_a_layernorm": "q_a_layernorm"
                    }
                    module.attention.fused_a.tllm_to_externel_key_dict = {
                        "fused_a": "kv_a_proj_with_mqa"
                    }  # v2 lite

                module.attention.kv_a_layernorm.tllm_to_externel_key_dict = {
                    'kv_a_layernorm': 'kv_a_layernorm'
                }

                if index > 0:

                    module.mlp.shared_expert.fc.tllm_to_externel_key_dict = {
                        "fc": ["up_proj", "gate_proj"],
                        "shared_expert": "shared_experts"
                    }
                    module.mlp.shared_expert.proj.tllm_to_externel_key_dict = {
                        "shared_expert": "shared_experts"
                    }
                    module.mlp.fc.tllm_to_externel_key_dict = {
                        "fc": [
                            f"experts.{expert}.up_proj"
                            for expert in rank_experts
                        ] + [
                            f"experts.{expert}.gate_proj"
                            for expert in rank_experts
                        ]
                    }
                    module.mlp.proj.tllm_to_externel_key_dict = {
                        "proj": [
                            f"experts.{expert}.down_proj"
                            for expert in rank_experts
                        ]
                    }
                    module.mlp.router.tllm_to_externel_key_dict = {
                        "mlp": "mlp",
                        "router": "gate"
                    }

            loader = ModelWeightsLoader(model_dir, custom_dict)
            loader.generate_tllm_weights(deepseek)
            return deepseek

        if use_safetensors_loading:
            weights = load_weights_from_hf_safetensors(
                model_dir,
                pretrained_config,
                mapping,
                use_parallel_embedding=pretrained_config.use_parallel_embedding,
                sharding_dim=pretrained_config.embedding_sharding_dim)
            deepseek.load(weights)
            return deepseek


# ---------------------------------------------------------------------------
# MTP (Multi-Token Prediction) speculative decoding for DeepSeek R1/V3
# ---------------------------------------------------------------------------

def _import_eagle_plugins():
    """Lazily import Eagle C++ plugin wrappers to avoid circular imports."""
    from ..eagle.model import (TreeParams,
                               eagle_draft_decoder_plugin,
                               eagle_prepare_drafter_inputs_plugin,
                               eagle_sample_and_accept_draft_plugin)
    return (TreeParams, eagle_sample_and_accept_draft_plugin,
            eagle_draft_decoder_plugin, eagle_prepare_drafter_inputs_plugin)


class DeepseekV2MTPForCausalLM(DeepseekV2ForCausalLM):
    """DeepSeek R1/V3 with MTP speculative decoding on the TRT backend.

    MTP uses a **single set of weights** (one MTP draft head in the checkpoint)
    run autoregressively N times to generate N draft tokens.

    To remain compatible with TRT-LLM's paged-KV-cache infrastructure
    (which requires a fixed ``local_layer_idx`` per attention plugin call),
    we create N ``DeepseekV2Attention`` objects with distinct KV-cache slots
    (base_num_hidden_layers + 0 … + N-1).  All non-attention components
    (enorm, hnorm, eh_proj, input_layernorm, post_layernorm, mlp) share a
    **single module instance** called N times in the forward loop.  The base
    model's ``vocab_embedding`` and ``lm_head`` are reused directly (weight
    tying).  All N attention objects load their weights from the same single
    MTP layer in the checkpoint at conversion time.
    """

    config_class = DeepseekV2MTPConfig

    def __init__(self, config: DeepseekV2MTPConfig):
        super().__init__(config)
        self._num_mtp_layers = config.num_nextn_predict_layers
        self._max_draft_len = config.num_nextn_predict_layers  # linear chain
        self._max_non_leaves_per_layer = 1  # linear chain

        if not self.mapping.is_last_pp_rank():
            return

        N = config.num_nextn_predict_layers

        # ------------------------------------------------------------------
        # Shared non-attention components (single instance, called N times)
        # ------------------------------------------------------------------
        self.mtp_enorm = RmsNorm(normalized_shape=config.hidden_size,
                                 eps=config.norm_epsilon,
                                 dtype=config.dtype)
        self.mtp_hnorm = RmsNorm(normalized_shape=config.hidden_size,
                                 eps=config.norm_epsilon,
                                 dtype=config.dtype)
        # concat([enorm(embed), hnorm(hidden)]) → hidden_size; not TP-split
        self.mtp_eh_proj = ColumnLinear(2 * config.hidden_size,
                                        config.hidden_size,
                                        bias=False,
                                        dtype=config.dtype,
                                        tp_group=None,
                                        tp_size=1,
                                        gather_output=True)
        self.mtp_input_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                           eps=config.norm_epsilon,
                                           dtype=config.dtype)
        self.mtp_post_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                          eps=config.norm_epsilon,
                                          dtype=config.dtype)
        # MTP head always uses a dense MLP (not MoE)
        self.mtp_mlp = GatedMLP(hidden_size=config.hidden_size,
                                 ffn_hidden_size=config.intermediate_size,
                                 hidden_act=non_gated_version(config.hidden_act),
                                 dtype=config.dtype,
                                 bias=False,
                                 tp_group=config.mapping.tp_group,
                                 tp_size=config.mapping.tp_size,
                                 quant_mode=config.quant_mode)

        # ------------------------------------------------------------------
        # N attention objects — one KV-cache slot per autoregressive step.
        # All N objects load their weights from the same single MTP layer in
        # the checkpoint; the distinct local_layer_idx values let the paged-KV
        # plugin track each step's KV history independently.
        # ------------------------------------------------------------------
        total_layers = config.num_hidden_layers + N
        layers_range = config.mapping.pp_layers(total_layers)
        self.mtp_attentions = ModuleList([
            DeepseekV2Attention(
                local_layer_idx=(config.num_hidden_layers + i) - layers_range[0],
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                max_position_embeddings=config.max_position_embeddings,
                eps=config.norm_epsilon,
                attention_mask_type=AttentionMaskType.causal,
                dtype=config.dtype,
                position_embedding_type=PositionEmbeddingType.learned_absolute,
                rotary_embedding_base=config.rotary_base,
                rotary_embedding_scaling=None,
                rotary_embedding_beta_fast=config.rotary_scaling['beta_fast'],
                rotary_embedding_beta_slow=config.rotary_scaling['beta_slow'],
                rotary_embedding_mscale=config.rotary_scaling['mscale'],
                rotary_embedding_mscale_all_dim=config.rotary_scaling[
                    'mscale_all_dim'],
                rotary_embedding_origin_max_position=config.rotary_scaling[
                    'original_max_position_embeddings'],
                rotary_scaling=config.rotary_scaling,
                tp_group=config.mapping.tp_group,
                tp_size=config.mapping.tp_size,
                tp_rank=config.mapping.tp_rank)
            for i in range(N)
        ])

    # ------------------------------------------------------------------
    # MTP forward step (shared weights, per-step attention)
    # ------------------------------------------------------------------

    def _forward_mtp_step(self, li: int, drafter_inputs: dict,
                          logits_dtype: str = 'float32'):
        """Run one MTP autoregressive step using the shared weight set.

        Non-attention ops (enorm/hnorm/eh_proj/mlp/layer-norms) are shared
        single-instance modules called for every step li.  self.mtp_attentions[li]
        provides a distinct KV-cache slot for step li so each autoregressive
        step's key/value history is tracked independently.  vocab_embedding and
        lm_head are inherited from the base model (weight tying).

        Returns:
            h_last: [num_selected, H] gathered hidden states for next step.
            logits: [num_selected, vocab_size] draft logits.
        """
        input_ids = drafter_inputs['input_ids']
        prev_hidden = drafter_inputs['hidden_states']
        last_token_indices = drafter_inputs['last_token_indices']
        kv_cache_params = drafter_inputs['kv_cache_params']
        attention_params = drafter_inputs['attention_params']
        spec_decoding_params = drafter_inputs['spec_decoding_params']

        # 1. Fuse embed(input_ids) + prev_hidden → projected hidden
        #    vocab_embedding is under self.transformer (weight tying with base model)
        embed = self.transformer.vocab_embedding(input_ids)
        x = self.mtp_eh_proj(
            concat([self.mtp_enorm(embed), self.mtp_hnorm(prev_hidden)],
                   dim=-1))

        # 2. Decoder layer — attention uses step-specific KV slot
        residual = x
        attn_out = self.mtp_attentions[li](
            hidden_states=self.mtp_input_layernorm(x),
            use_cache=True,
            spec_decoding_params=spec_decoding_params,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        h = residual + attn_out

        residual = h
        h = residual + self.mtp_mlp(self.mtp_post_layernorm(h))

        # 3. Draft logits for selected positions; lm_head shared with base model
        h_last = gather_last_token_logits(
            h, last_token_indices,
            default_net().plugin_config.remove_input_padding)
        logits = cast(self.lm_head(h_last), logits_dtype)
        return h_last, logits

    # ------------------------------------------------------------------
    # Internal helpers (mirrors EagleForCausalLM._prepare_drafter_inputs)
    # ------------------------------------------------------------------

    def _prepare_drafter_inputs(
            self, layer_idx, input_ids, chunked_context_next_tokens,
            accepted_token_ids, accepted_lens, accepted_path_ids,
            next_draft_tokens, next_draft_lens, next_draft_paths,
            prev_draft_lens, prev_draft_paths, input_attention_params,
            input_kv_cache_params, hidden_states,
            host_ctx_eagle_net_request_types,
            host_ctx_eagle_net_context_lengths,
            host_ctx_eagle_net_past_key_value_lengths,
            host_gen_eagle_net_request_types,
            host_gen_eagle_net_context_lengths,
            host_gen_eagle_net_past_key_value_lengths,
            hidden_size_batch_level_starts, input_gen_tokens,
            input_spec_decoding_generation_lengths, spec_decoding_use):

        (_, _, eagle_draft_decoder_plugin,
         eagle_prepare_drafter_inputs_plugin) = _import_eagle_plugins()

        drafter_inputs = eagle_prepare_drafter_inputs_plugin(
            layer_idx, self._num_mtp_layers, self._max_non_leaves_per_layer,
            input_attention_params, input_ids, chunked_context_next_tokens,
            accepted_token_ids, accepted_lens, accepted_path_ids,
            next_draft_tokens, next_draft_lens, next_draft_paths,
            prev_draft_lens, prev_draft_paths, hidden_size_batch_level_starts,
            input_gen_tokens, input_spec_decoding_generation_lengths)

        (sequence_length, context_lengths,
         spec_decoding_generation_lengths, spec_decoding_position_offsets,
         spec_decoding_packed_mask, output_ids, position_ids,
         hidden_states_indices, last_token_indices, num_last_token_indices,
         out_hidden_size_batch_level_starts) = drafter_inputs

        attention_params = input_attention_params
        kv_cache_params = input_kv_cache_params
        attention_params.sequence_length = sequence_length
        attention_params.context_lengths = context_lengths

        if layer_idx == 0:
            attention_params.host_request_types = \
                host_ctx_eagle_net_request_types
            attention_params.host_context_lengths = \
                host_ctx_eagle_net_context_lengths
            kv_cache_params.host_past_key_value_lengths = \
                host_ctx_eagle_net_past_key_value_lengths
        else:
            attention_params.host_request_types = \
                host_gen_eagle_net_request_types
            attention_params.host_context_lengths = \
                host_gen_eagle_net_context_lengths
            kv_cache_params.host_past_key_value_lengths = \
                host_gen_eagle_net_past_key_value_lengths

        spec_decoding_params = None
        if layer_idx > 0:
            spec_decoding_params = SpecDecodingParams(
                True, self._max_draft_len,
                spec_decoding_generation_lengths,
                spec_decoding_position_offsets,
                spec_decoding_packed_mask,
                spec_decoding_use)

        # Select hidden states at the positions required by the drafter
        hidden_states = index_select(hidden_states, 0, hidden_states_indices)
        hidden_states = hidden_states.view(
            concat([shape(hidden_states_indices, 0),
                    shape(hidden_states, 1)]),
            zero_is_placeholder=False)

        net_inputs = {
            'input_ids': output_ids,
            'position_ids': position_ids,
            'last_token_indices': last_token_indices,
            'attention_params': attention_params,
            'kv_cache_params': kv_cache_params,
            'spec_decoding_params': spec_decoding_params,
            'hidden_states': hidden_states,
        }
        return net_inputs, out_hidden_size_batch_level_starts, \
            num_last_token_indices

    def _mtp_fwd_helper(self, lm_logits, hidden_states, *args, **kwargs):
        """Accept previous draft tokens, then generate new draft tokens.

        Mirrors ``EagleForCausalLM._eagle_fwd_helper`` but runs MTP layers
        (with ``eh_proj`` fusion) instead of separate LLaMA draft models.
        """
        (TreeParams, eagle_sample_and_accept_draft_plugin,
         eagle_draft_decoder_plugin, _) = _import_eagle_plugins()

        input_tree_params = kwargs['tree_params']
        draft_tokens = kwargs['draft_tokens']
        draft_lens = kwargs['draft_lens']
        eagle_temperature = kwargs['eagle_temperature']
        rand_data_validation = kwargs['rand_data_validation']
        posterior_alpha = kwargs['posterior_alpha']
        posterior_threshold = kwargs['posterior_threshold']
        input_ids = kwargs['input_ids']
        chunked_context_next_tokens = kwargs['chunked_context_next_tokens']
        host_ctx_eagle_net_request_types = \
            kwargs['host_ctx_eagle_net_request_types']
        host_ctx_eagle_net_context_lengths = \
            kwargs['host_ctx_eagle_net_context_lengths']
        host_ctx_eagle_net_past_key_value_lengths = \
            kwargs['host_ctx_eagle_net_past_key_value_lengths']
        host_gen_eagle_net_request_types = \
            kwargs['host_gen_eagle_net_request_types']
        host_gen_eagle_net_context_lengths = \
            kwargs['host_gen_eagle_net_context_lengths']
        host_gen_eagle_net_past_key_value_lengths = \
            kwargs['host_gen_eagle_net_past_key_value_lengths']
        input_gen_tokens = kwargs['input_gen_tokens']
        greedy_sampling = kwargs['greedy_sampling']
        use_dynamic_tree = kwargs['use_dynamic_tree']
        dynamic_tree_max_topK = kwargs['dynamic_tree_max_topK']
        prev_scores = kwargs['prev_scores']
        current_expand_indices = kwargs['current_expand_indices']
        all_layers_scores = kwargs['all_layers_scores']
        all_layers_draft_token_ids = kwargs['all_layers_draft_token_ids']
        all_layers_draft_token_ids_predecessor = \
            kwargs['all_layers_draft_token_ids_predecessor']

        # ------------------------------------------------------------------
        # Step 1: Accept draft tokens from previous iteration
        # ------------------------------------------------------------------
        output = eagle_sample_and_accept_draft_plugin(
            lm_logits, draft_tokens, draft_lens, eagle_temperature,
            rand_data_validation, posterior_alpha, posterior_threshold,
            input_tree_params, greedy_sampling, use_dynamic_tree)
        (accepted_tokens, num_accepted_tokens, accepted_paths,
         next_draft_tokens, next_draft_lens, next_draft_paths,
         hidden_size_batch_level_starts) = output

        attention_params = kwargs['attention_params']
        kv_cache_params = kwargs['kv_cache_params']
        spec_decoding_params = kwargs['spec_decoding_params']

        input_hidden_states = hidden_states

        # ------------------------------------------------------------------
        # Step 2: Run each MTP head to generate the next draft tokens
        # ------------------------------------------------------------------
        for li in range(self._num_mtp_layers):
            drafter_inputs, hidden_size_batch_level_starts, \
                num_last_token_indices = self._prepare_drafter_inputs(
                    layer_idx=li,
                    input_ids=input_ids,
                    chunked_context_next_tokens=chunked_context_next_tokens,
                    accepted_token_ids=accepted_tokens,
                    accepted_lens=num_accepted_tokens,
                    accepted_path_ids=accepted_paths,
                    next_draft_tokens=next_draft_tokens,
                    next_draft_lens=next_draft_lens,
                    next_draft_paths=next_draft_paths,
                    prev_draft_lens=draft_lens,
                    prev_draft_paths=input_tree_params.paths,
                    input_attention_params=attention_params,
                    input_kv_cache_params=kv_cache_params,
                    hidden_states=input_hidden_states,
                    host_ctx_eagle_net_request_types=
                    host_ctx_eagle_net_request_types,
                    host_ctx_eagle_net_context_lengths=
                    host_ctx_eagle_net_context_lengths,
                    host_ctx_eagle_net_past_key_value_lengths=
                    host_ctx_eagle_net_past_key_value_lengths,
                    host_gen_eagle_net_request_types=
                    host_gen_eagle_net_request_types,
                    host_gen_eagle_net_context_lengths=
                    host_gen_eagle_net_context_lengths,
                    host_gen_eagle_net_past_key_value_lengths=
                    host_gen_eagle_net_past_key_value_lengths,
                    hidden_size_batch_level_starts=
                    hidden_size_batch_level_starts,
                    input_gen_tokens=input_gen_tokens,
                    input_spec_decoding_generation_lengths=
                    spec_decoding_params.spec_decoding_generation_lengths,
                    spec_decoding_use=spec_decoding_params.spec_decoding_use)

            # Run one MTP autoregressive step: shared weights + per-step attention
            h_out, draft_logits = self._forward_mtp_step(
                li, drafter_inputs, logits_dtype=self.config.logits_dtype)

            # Decode draft tokens via Eagle plugin
            top_k_sampling = True  # TODO: make configurable
            (next_draft_tokens, next_draft_lens, next_draft_paths,
             prev_scores, current_expand_indices,
             all_layers_scores, all_layers_draft_token_ids,
             all_layers_draft_token_ids_predecessor) = \
                eagle_draft_decoder_plugin(
                    li, self._num_mtp_layers, top_k_sampling,
                    draft_logits, num_last_token_indices, next_draft_paths,
                    use_dynamic_tree, dynamic_tree_max_topK,
                    next_draft_tokens, next_draft_lens,
                    prev_scores, current_expand_indices,
                    all_layers_scores, all_layers_draft_token_ids,
                    all_layers_draft_token_ids_predecessor)

            # Update hidden-state buffer for the next MTP layer
            if li == 0:
                eagle_net_0_sequence_length = \
                    drafter_inputs['attention_params'].sequence_length
                input_hidden_states = h_out
            else:
                input_hidden_states = concat([input_hidden_states, h_out])

            kv_cache_params = drafter_inputs['kv_cache_params']
            attention_params = drafter_inputs['attention_params']
            attention_params.context_lengths = eagle_net_0_sequence_length
            attention_params.sequence_length = eagle_net_0_sequence_length

        # Mark speculative-decoding outputs
        accepted_tokens.mark_output('accepted_tokens')
        num_accepted_tokens.mark_output('num_accepted_tokens')
        accepted_paths.mark_output('accepted_paths')
        next_draft_tokens.mark_output('next_draft_tokens')
        next_draft_lens.mark_output('next_draft_lens')
        next_draft_paths.mark_output('next_draft_paths')

        return next_draft_tokens

    # ------------------------------------------------------------------
    # forward / prepare_inputs
    # ------------------------------------------------------------------

    # Extra kwargs consumed by MTP logic; everything else goes to the base
    # model forward.
    _MTP_EXTRA_ARGS = [
        'draft_tokens', 'draft_lens', 'eagle_temperature',
        'rand_data_validation', 'tree_params',
        'host_ctx_eagle_net_request_types',
        'host_ctx_eagle_net_context_lengths',
        'host_ctx_eagle_net_past_key_value_lengths',
        'host_gen_eagle_net_request_types',
        'host_gen_eagle_net_context_lengths',
        'host_gen_eagle_net_past_key_value_lengths',
        'input_gen_tokens', 'chunked_context_next_tokens',
        'posterior_alpha', 'posterior_threshold', 'greedy_sampling',
        'use_dynamic_tree', 'dynamic_tree_max_topK',
        'prev_scores', 'current_expand_indices',
        'all_layers_scores', 'all_layers_draft_token_ids',
        'all_layers_draft_token_ids_predecessor',
    ]

    def forward(self, *args, **kwargs):
        extra = set(self._MTP_EXTRA_ARGS)
        base_kwargs = {k: v for k, v in kwargs.items() if k not in extra}

        # Base model forward → (lm_logits, gathered_hidden, all_hidden)
        # on the last PP rank, or hidden_states on other ranks.
        result = super().forward(*args, **base_kwargs)

        if self.mapping.is_last_pp_rank():
            # Remove 'hidden_states' from kwargs to avoid confusion with
            # the PP-send path (mirrors Eagle implementation).
            kwargs = {k: v for k, v in kwargs.items() if k != 'hidden_states'}

        if self.mapping.is_last_pp_rank():
            assert kwargs['use_cache'] and \
                default_net().plugin_config.paged_kv_cache, \
                'DeepseekV2MTPForCausalLM requires use_cache=True and paged KV cache'
            lm_logits, _gathered_hidden, all_hidden_states = result
            lm_logits = cast(lm_logits, self.config.logits_dtype)
            next_draft_tokens = self._mtp_fwd_helper(
                lm_logits, all_hidden_states, *args, **kwargs)
            return next_draft_tokens
        else:
            result.mark_output('hidden_states_output', self.config.dtype)
            return result

    def prepare_inputs(self, *args, **kwargs):
        from ..generation_mixin import GenerationMixin

        kwargs['speculative_decoding_draft_tokens_external'] = False
        kwargs['max_draft_len'] = self._max_draft_len
        kwargs['spec_decoding_is_generation_length_variable'] = True
        # Allocate KV-cache slots for base layers + MTP layers (1 per head)
        kwargs['num_hidden_layers'] = (self.config.num_hidden_layers +
                                       self._num_mtp_layers)

        inputs = super().prepare_inputs(*args, **kwargs)

        assert inputs.get('spec_decoding_params') is not None, (
            'prepare_inputs must be called with speculative-decoding settings')

        # ------------------------------------------------------------------
        # Add Eagle-compatible input tensors (same as EagleForCausalLM)
        # ------------------------------------------------------------------
        remove_input_padding = \
            default_net().plugin_config.remove_input_padding
        use_gpt_attention_plugin = \
            default_net().plugin_config.gpt_attention_plugin
        use_gemm_plugin = default_net().plugin_config.gemm_plugin
        paged_kv_cache = default_net().plugin_config.paged_kv_cache
        multiple_profiles = default_net().plugin_config.multiple_profiles
        max_batch_size = kwargs['max_batch_size']

        kv_cache_type = (KVCacheType.PAGED
                         if paged_kv_cache else KVCacheType.CONTINUOUS)
        enable_ctx_gen_opt_profiles = GenerationMixin.has_ctx_gen_opt_profiles(
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            use_gemm_plugin=use_gemm_plugin,
            remove_input_padding=remove_input_padding,
            kv_cache_type=kv_cache_type)

        _, ranges = GenerationMixin.get_profiles_ranges(
            max_batch_size=max_batch_size,
            max_beam_width=kwargs['max_beam_width'],
            max_input_len=kwargs['max_input_len'],
            max_num_tokens=kwargs['max_num_tokens'],
            max_draft_len=self._max_draft_len,
            opt_batch_size=kwargs.get('opt_batch_size'),
            opt_num_tokens=kwargs.get('opt_num_tokens'),
            enable_ctx_gen_opt_profiles=enable_ctx_gen_opt_profiles,
            multiple_profiles=multiple_profiles,
            kv_cache_type=kv_cache_type)

        bb_range = ranges['bb_range']
        gt_range = GenerationMixin.default_range(
            max_batch_size * (self._max_draft_len + 1), min_range=0,
            opt_offset=1)

        draft_len_range = [self._max_draft_len] * len(bb_range)
        decoding_len_range = [self._max_draft_len + 1] * len(bb_range)
        path_len_range = [self._num_mtp_layers + 1] * len(bb_range)
        gen_tokens_range = [gt_range] * len(bb_range)
        num_eagle_layers_range = [self._num_mtp_layers] * len(bb_range)
        draft_len_sq_range = [
            self._max_draft_len * self._max_draft_len] * len(bb_range)

        draft_tokens = Tensor(
            name='draft_tokens', dtype=trt.int32,
            shape=[-1, self._max_draft_len],
            dim_range=OrderedDict([('batch_size', bb_range),
                                   ('draft_len', draft_len_range)]))
        draft_lens = Tensor(
            name='draft_lens', dtype=trt.int32, shape=[-1],
            dim_range=OrderedDict([('batch_size', bb_range)]))
        eagle_temperature = Tensor(
            name='eagle_temperature', dtype=trt.float32, shape=[-1],
            dim_range=OrderedDict([('batch_size', bb_range)]))
        rand_data_validation = Tensor(
            name='rand_data_validation', dtype=trt.float32,
            shape=[-1, self._max_draft_len + 1],
            dim_range=OrderedDict([('batch_size', bb_range),
                                   ('decoding_len', decoding_len_range)]))
        posterior_alpha = Tensor(
            name='posterior_alpha', dtype=trt.float32, shape=[-1],
            dim_range=OrderedDict([('batch_size', bb_range)]))
        posterior_threshold = Tensor(
            name='posterior_threshold', dtype=trt.float32, shape=[-1],
            dim_range=OrderedDict([('batch_size', bb_range)]))
        draft_paths = Tensor(
            name='draft_paths', dtype=trt.int32,
            shape=[-1, self._max_draft_len + 1, self._num_mtp_layers + 1],
            dim_range=OrderedDict([('batch_size', bb_range),
                                   ('decoding_len', decoding_len_range),
                                   ('path_len', path_len_range)]))

        host_ctx_eagle_net_request_types = Tensor(
            name='host_ctx_eagle_net_request_types', dtype=trt.int32,
            shape=[-1], dim_range=OrderedDict([('batch_size', bb_range)]))
        host_ctx_eagle_net_context_lengths = Tensor(
            name='host_ctx_eagle_net_context_lengths', dtype=trt.int32,
            shape=[-1], dim_range=OrderedDict([('batch_size', bb_range)]))
        host_ctx_eagle_net_past_key_value_lengths = Tensor(
            name='host_ctx_eagle_net_past_key_value_lengths', dtype=trt.int32,
            shape=[-1], dim_range=OrderedDict([('batch_size', bb_range)]))
        host_gen_eagle_net_request_types = Tensor(
            name='host_gen_eagle_net_request_types', dtype=trt.int32,
            shape=[-1], dim_range=OrderedDict([('batch_size', bb_range)]))
        host_gen_eagle_net_context_lengths = Tensor(
            name='host_gen_eagle_net_context_lengths', dtype=trt.int32,
            shape=[-1], dim_range=OrderedDict([('batch_size', bb_range)]))
        host_gen_eagle_net_past_key_value_lengths = Tensor(
            name='host_gen_eagle_net_past_key_value_lengths', dtype=trt.int32,
            shape=[-1], dim_range=OrderedDict([('batch_size', bb_range)]))

        input_gen_tokens = Tensor(
            name='input_gen_tokens', dtype=trt.int32, shape=[-1],
            dim_range=OrderedDict([('gen_tokens', gen_tokens_range)]))
        chunked_context_next_tokens = Tensor(
            name='chunked_context_next_tokens', dtype=trt.int32, shape=[-1],
            dim_range=OrderedDict([('batch_size', bb_range)]))
        greedy_sampling = Tensor(
            name='greedy_sampling', dtype=trt.int32, shape=[1])
        use_dynamic_tree = Tensor(
            name='use_dynamic_tree', dtype=trt.int32, shape=[1])
        dynamic_tree_max_topK = Tensor(
            name='dynamic_tree_max_topK', dtype=trt.int32, shape=[1])
        prev_scores = Tensor(
            name='prev_scores', dtype=trt.float32,
            shape=[-1, self._max_draft_len],
            dim_range=OrderedDict([('batch_size', bb_range),
                                   ('draft_len', draft_len_range)]))
        current_expand_indices = Tensor(
            name='current_expand_indices', dtype=trt.int32,
            shape=[-1, self._max_draft_len],
            dim_range=OrderedDict([('batch_size', bb_range),
                                   ('draft_len', draft_len_range)]))
        all_layers_scores = Tensor(
            name='all_layers_scores', dtype=trt.float32,
            shape=[-1, self._num_mtp_layers,
                   self._max_draft_len * self._max_draft_len],
            dim_range=OrderedDict([
                ('batch_size', bb_range),
                ('num_eagle_layers', num_eagle_layers_range),
                ('draft_len_square', draft_len_sq_range)]))
        all_layers_draft_token_ids = Tensor(
            name='all_layers_draft_token_ids', dtype=trt.int32,
            shape=[-1, self._num_mtp_layers,
                   self._max_draft_len * self._max_draft_len],
            dim_range=OrderedDict([
                ('batch_size', bb_range),
                ('num_eagle_layers', num_eagle_layers_range),
                ('draft_len_square', draft_len_sq_range)]))
        all_layers_draft_token_ids_predecessor = Tensor(
            name='all_layers_draft_token_ids_predecessor', dtype=trt.int32,
            shape=[-1, self._num_mtp_layers,
                   self._max_draft_len * self._max_draft_len],
            dim_range=OrderedDict([
                ('batch_size', bb_range),
                ('num_eagle_layers', num_eagle_layers_range),
                ('draft_len_square', draft_len_sq_range)]))

        from ..eagle.model import TreeParams
        tree_params = TreeParams(paths=draft_paths)

        inputs['draft_tokens'] = draft_tokens
        inputs['draft_lens'] = draft_lens
        inputs['eagle_temperature'] = eagle_temperature
        inputs['posterior_alpha'] = posterior_alpha
        inputs['posterior_threshold'] = posterior_threshold
        inputs['rand_data_validation'] = rand_data_validation
        inputs['tree_params'] = tree_params
        inputs['host_ctx_eagle_net_request_types'] = \
            host_ctx_eagle_net_request_types
        inputs['host_ctx_eagle_net_context_lengths'] = \
            host_ctx_eagle_net_context_lengths
        inputs['host_ctx_eagle_net_past_key_value_lengths'] = \
            host_ctx_eagle_net_past_key_value_lengths
        inputs['host_gen_eagle_net_request_types'] = \
            host_gen_eagle_net_request_types
        inputs['host_gen_eagle_net_context_lengths'] = \
            host_gen_eagle_net_context_lengths
        inputs['host_gen_eagle_net_past_key_value_lengths'] = \
            host_gen_eagle_net_past_key_value_lengths
        inputs['input_gen_tokens'] = input_gen_tokens
        inputs['chunked_context_next_tokens'] = chunked_context_next_tokens
        inputs['greedy_sampling'] = greedy_sampling
        inputs['use_dynamic_tree'] = use_dynamic_tree
        inputs['dynamic_tree_max_topK'] = dynamic_tree_max_topK
        inputs['prev_scores'] = prev_scores
        inputs['current_expand_indices'] = current_expand_indices
        inputs['all_layers_scores'] = all_layers_scores
        inputs['all_layers_draft_token_ids'] = all_layers_draft_token_ids
        inputs['all_layers_draft_token_ids_predecessor'] = \
            all_layers_draft_token_ids_predecessor

        return inputs
