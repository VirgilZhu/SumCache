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


class LlamaSumCacheModel(LlamaModel):
    def __init__(self, config: LlamaConfig, num_sum_tokens=2):
        super().__init__(config)
        ### SumCache Add Below ###
        self.num_sum_tokens = num_sum_tokens
        self.sum_token_ids = torch.arange(config.vocab_size, config.vocab_size + num_sum_tokens)
        self.vocab_size = config.vocab_size + num_sum_tokens
        
        # 扩展嵌入层以包含sum tokens
        old_embed_tokens = self.embed_tokens
        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)
        
        # 复制原始权重并初始化sum tokens
        with torch.no_grad():
            self.embed_tokens.weight[:config.vocab_size] = old_embed_tokens.weight
            # 使用平均嵌入初始化sum tokens
            avg_embed = old_embed_tokens.weight.mean(dim=0)
            for i in range(config.vocab_size, self.vocab_size):
                self.embed_tokens.weight[i] = avg_embed
        ### SumCache Add Above ###

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
        sum_token_positions: Optional[torch.Tensor] = None
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
        if sum_token_positions is not None and sum_token_positions.numel() > 0:
            # 找到最后一个sum_token的位置
            last_sum_token = sum_token_positions.max().item()
            
            # 创建sum_token屏蔽矩阵
            sum_token_mask = torch.ones((sequence_length, target_length), dtype=dtype, device=device)
            sum_token_mask[:, :last_sum_token] = min_dtype
            
            # 合并到因果掩码
            causal_mask = torch.min(causal_mask, sum_token_mask)
        ### SumCache ADD Above ###

        return causal_mask


class SumKVCache_LayerWise:
    def __init__(
        self,
        sum_cache_size=256,
        recent_size=256,
        chunk_size=26,
        topk_keep=4,
        num_sum_tokens=2,
        k_seq_dim=2,
        v_seq_dim=2,
        head_dim=128,
    ):
        self.sum_cache_size = sum_cache_size
        self.recent_size = recent_size
        self.cache_max_size = sum_cache_size + recent_size
        self.chunk_size = chunk_size
        self.topk_keep = topk_keep
        self.num_sum_tokens = num_sum_tokens
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        
        self.importance_scores = None
        self.last_compressed_idx = 0

        self.summary_q = nn.Parameter(torch.randn(num_sum_tokens, head_dim))

    def __call__(self, key_states, value_states, layer_idx, past_key_values, attn_weights):
        """
        直接修改past_key_values，避免复制缓存
        """
        bsz, num_heads, seq_len, head_dim = key_states.shape
        
        # 初始化重要性分数
        if self.importance_scores is None:
            self.importance_scores = torch.zeros(bsz, 0, device=key_states.device)
            
        # 更新重要性分数：注意力权重列求和
        self._update_importance_scores(attn_weights)

        # 正确计算当前总长度
        current_kv_len = past_key_values.key_cache[layer_idx].shape[2] if past_key_values else 0
        total_len = current_kv_len + seq_len if seq_len == 1 else current_kv_len

        # 如果需要压缩的长度大于0，则循环压缩所有完整chunk
        while (total_len > self.recent_size and
               (total_len - self.recent_size - self.last_compressed_idx) >= self.chunk_size):
            past_key_values = self.compress_chunk(
                key_states, value_states, layer_idx, past_key_values, total_len
            )

            # 更新缓存长度（压缩后长度会变化）
            current_kv_len = past_key_values.key_cache[layer_idx].shape[2]
            total_len = current_kv_len + seq_len if seq_len == 1 else current_kv_len

        return past_key_values
    
    def _update_importance_scores(self, attn_weights):
        """更新token重要性分数（注意力权重列求和）"""
        # attn_weights形状: [bsz, num_heads, q_len, kv_len]
        new_scores = attn_weights.sum(dim=1).sum(dim=1)  # 按头和批次求和
        seq_len = new_scores.shape[-1]
        
        if self.importance_scores.shape[1] < seq_len:
            # 扩展分数矩阵
            padding = seq_len - self.importance_scores.shape[1]
            self.importance_scores = F.pad(self.importance_scores, (0, padding))
        
        self.importance_scores[:, -seq_len:] += new_scores
    
    def compress_chunk(self, key_states, value_states, layer_idx, past_key_values, total_len):
        # 计算要压缩的chunk范围
        start_idx = self.last_compressed_idx
        end_idx = min(start_idx + self.chunk_size, total_len - self.recent_size)
        chunk_size = end_idx - start_idx

        # 从 Cache 中获取chunk
        key_chunk = past_key_values.key_cache[layer_idx][:, :, start_idx:end_idx]
        value_chunk = past_key_values.value_cache[layer_idx][:, :, start_idx:end_idx]
        scores_chunk = self.importance_scores[:, start_idx:end_idx]
        
        bsz, num_heads, chunk_len, head_dim = key_chunk.shape
        
        # top-k important tokens
        _, topk_indices = torch.topk(scores_chunk, k=min(self.topk_keep, chunk_len), dim=-1)
        topk_indices = topk_indices.sort(dim=-1).values
        topk_indices_exp = topk_indices.view(bsz, 1, -1, 1).expand(-1, num_heads, -1, head_dim)

        key_topk = torch.gather(key_chunk, dim=2, index=topk_indices_exp)
        value_topk = torch.gather(value_chunk, dim=2, index=topk_indices_exp)
        
        # 可学习的summary query向量
        summary_q = self.summary_q.to(device=key_chunk.device)
        summary_q = summary_q.unsqueeze(0).unsqueeze(0)  # [1,1,num_sum,head_dim]
        summary_q = summary_q.expand(bsz, num_heads, -1, -1)

        # 计算summary tokens的注意力分数
        attn_scores = torch.matmul(             # [bsz, num_heads, num_sum_tokens, chunk_len]
            summary_q,
            key_chunk.transpose(-1, -2)
        ) / math.sqrt(head_dim)

        summary_important_scores = attn_scores.sum(dim=-1).sum(dim=1)
        summary_scores = summary_important_scores

        # 应用softmax获取注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 使用注意力权重聚合key
        summary_k = torch.matmul(attn_weights, key_chunk)
        # 使用注意力权重聚合value
        summary_v = torch.matmul(attn_weights, value_chunk)
        
        # 组合重要token和summary tokens
        compressed_key = torch.cat([key_topk, summary_k], dim=2)
        compressed_value = torch.cat([value_topk, summary_v], dim=2)
        
        # 更新缓存
        if past_key_values is None:
            # 创建新缓存
            past_key_values = DynamicCache()
            past_key_values.key_cache = [compressed_key]
            past_key_values.value_cache = [compressed_value]
            
            # 添加剩余部分（如果有）
            if chunk_len < key_states.shape[2]:
                remaining_key = key_states[:, :, chunk_size:]
                remaining_value = value_states[:, :, chunk_size:]
                past_key_values.update(remaining_key, remaining_value, layer_idx)
        else:
            # 替换压缩的chunk
            new_key_cache = torch.cat([
                past_key_values.key_cache[layer_idx][:, :, :start_idx],
                compressed_key,
                past_key_values.key_cache[layer_idx][:, :, end_idx:]
            ], dim=2)
            
            new_value_cache = torch.cat([
                past_key_values.value_cache[layer_idx][:, :, :start_idx],
                compressed_value,
                past_key_values.value_cache[layer_idx][:, :, end_idx:]
            ], dim=2)
            
            past_key_values.key_cache[layer_idx] = new_key_cache
            past_key_values.value_cache[layer_idx] = new_value_cache
        
        # 更新重要性分数（移除压缩部分，添加新token分数）
        topk_scores = torch.gather(scores_chunk, 1, topk_indices)

        new_score_segment = torch.cat([topk_scores, summary_scores], dim=1)

        self.importance_scores = torch.cat([
            self.importance_scores[:, :start_idx],
            new_score_segment,
            self.importance_scores[:, end_idx:]
        ], dim=1)

        self.last_compressed_idx = start_idx + new_score_segment.size(1)
        
        return past_key_values


class LlamaSumCacheAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        head_dim = config.hidden_size // config.num_attention_heads
        self.kv_cache = SumKVCache_LayerWise(
            sum_cache_size=2048,
            recent_size=2048,
            chunk_size=26,
            topk_keep=4,
            num_sum_tokens=2,
            k_seq_dim=2,
            v_seq_dim=2,
            head_dim=head_dim,
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
        sum_token_positions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # if past_key_value is not None:
        #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

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

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        ### SumCache Add Below ###
        past_key_value = self.kv_cache(
            key_states, value_states, self.layer_idx, 
            past_key_value, attn_weights.detach()
        )
        ### SumCache Add Above ###

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    

class LlamaSumCacheForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaSumCacheModel(config)
        self.vocab_size = config.vocab_size + self.model.num_sum_tokens

        # 扩展lm_head以包含sum tokens
        old_lm_head = self.lm_head
        self.lm_head = nn.Linear(config.hidden_size, self.model.vocab_size, bias=False)
        with torch.no_grad():
            self.lm_head.weight[:old_lm_head.weight.size(0)] = old_lm_head.weight
            # 初始化sum token的输出权重
            avg_weight = old_lm_head.weight.mean(dim=0)
            for i in range(old_lm_head.weight.size(0), self.vocab_size):
                self.lm_head.weight[i] = avg_weight

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
        sum_token_positions: Optional[torch.Tensor] = None,  # 新增sum_token位置
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 修改causal_mask生成以包含sum_token_positions
        causal_mask = self.model._update_causal_mask(
            attention_mask, inputs_embeds, cache_position,
            past_key_values, output_attentions, sum_token_positions
        )
        
        # 将sum_token_positions传递给decoder layers
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            sum_token_positions=sum_token_positions,
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