# transformers.__version__ == '4.43.3'
import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaModel, LlamaForCausalLM, apply_rotary_pos_emb, repeat_kv
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging

logger = logging.get_logger(__name__)

from .utils_sumcache import SumKVCache_LayerWise


class LlamaSumCacheAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.kv_cache = SumKVCache_LayerWise(
            compressed_cache_limit=4096-32,
            recent_size=32,
            chunk_size=64,
            topk_important=1,
            num_sum_tokens=2,
            k_seq_dim=2,
            v_seq_dim=2,
            sum_compress_ratio=0.5,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        # sum_token_positions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # [H2O] update below
        kv_seq_len = key_states.shape[-2]
        if kv_seq_len == 1 and self.layer_idx == 0:
            position_ids += self.position_ids_margin
        else:
            self.position_ids_margin = 0
        # [H2O] update above

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        ### SumCache Add Below ###
        if past_key_value is not None and q_len > 1:
            self.kv_cache._clean_scores()
            past_key_value = self.kv_cache(
                key_states, value_states, self.layer_idx,
                past_key_value, attn_weights.detach().clone(),
                rotary_emb=self.rotary_emb,
                num_key_value_groups=1
            )
            self.position_ids_margin = kv_seq_len - self.kv_cache.cache_size
        ### SumCache Add Above ###

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaSumCacheModel(LlamaModel):
    def __init__(self, config: LlamaConfig, num_sum_tokens=2):
        super().__init__(config)
        ### SumCache Add Below ###
        self.num_sum_tokens = num_sum_tokens
        # self.sum_token_ids = torch.arange(config.vocab_size, config.vocab_size + num_sum_tokens)
        # self.vocab_size = config.vocab_size + num_sum_tokens
        
        # 扩展嵌入层以包含sum tokens
        # old_embed_tokens = self.embed_tokens
        # self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)
        
        # 复制原始权重并初始化sum tokens
        # with torch.no_grad():
        #     self.embed_tokens.weight[:config.vocab_size] = old_embed_tokens.weight
            # 使用平均嵌入初始化sum tokens
            # avg_embed = old_embed_tokens.weight.mean(dim=0)
            # for i in range(config.vocab_size, self.vocab_size):
            #     self.embed_tokens.weight[i] = avg_embed
        ### SumCache Add Above ###

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
        # sum_token_positions: Optional[torch.Tensor] = None
    ):

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        ### SumCache ADD Below ###
        # if sum_token_positions is not None and sum_token_positions.numel() > 0:
        #     # 找到最后一个sum_token的位置
        #     last_sum_token = sum_token_positions.max().item()
        #
        #     # 创建sum_token屏蔽矩阵
        #     sum_token_mask = torch.ones((sequence_length, target_length), dtype=dtype, device=device)
        #     sum_token_mask[:, :last_sum_token] = min_dtype
        #
        #     # 合并到因果掩码
        #     causal_mask = torch.min(causal_mask, sum_token_mask)
        ### SumCache ADD Above ###

        return causal_mask
    

class LlamaSumCacheForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaSumCacheModel(config)
        # self.vocab_size = config.vocab_size + self.model.num_sum_tokens

        # 扩展lm_head以包含sum tokens
        # old_lm_head = self.lm_head
        # self.lm_head = nn.Linear(config.hidden_size, self.model.vocab_size, bias=False)
        # with torch.no_grad():
        #     self.lm_head.weight[:old_lm_head.weight.size(0)] = old_lm_head.weight
            # 初始化sum token的输出权重
            # avg_weight = old_lm_head.weight.mean(dim=0)
            # for i in range(old_lm_head.weight.size(0), self.vocab_size):
            #     self.lm_head.weight[i] = avg_weight

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_last_logits_only: bool = False,
        # sum_token_positions: Optional[torch.Tensor] = None,  # 新增sum_token位置
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 修改causal_mask生成以包含sum_token_positions
        causal_mask = self.model._update_causal_mask(
            attention_mask, inputs_embeds, cache_position,
            past_key_values, output_attentions
            # past_key_values, output_attentions, sum_token_positions
        )
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            # sum_token_positions=sum_token_positions,
        )

        hidden_states = outputs[0]
        if output_last_logits_only:
            logits = self.lm_head(hidden_states[:,-1:,:])
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()
        loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )