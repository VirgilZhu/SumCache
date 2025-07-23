import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

# reverse opertaion of repeat_kv
def sum_group(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states.reshape(
        batch, num_heads // n_rep, n_rep, slen, head_dim)
    return hidden_states.sum(2)


class SumKVCache_LayerWise:
    def __init__(
        self,
        sum_cache_size=4,
        recent_size=512,
        chunk_size=26,
        topk_important=4,
        num_sum_tokens=2,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        self.sum_cache_size = sum_cache_size
        self.recent_size = recent_size
        self.cache_max_size = sum_cache_size + recent_size
        self.chunk_size = chunk_size
        self.cache_size = 0
        self.topk_important = topk_important
        self.num_sum_tokens = num_sum_tokens
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        
        self.importance_scores = None
        self.last_compressed_idx = 0

        self.summary_q = None

    def __call__(self, key_states, value_states, layer_idx, past_key_values, attn_weights, rotary_emb, num_key_value_groups):
        self._update_importance_scores(attn_weights, num_key_value_groups)

        _, num_heads, seq_len, head_dim = key_states.shape

        if self.summary_q is None:
            self.summary_q = nn.Parameter(torch.randn(num_heads, self.num_sum_tokens, head_dim, dtype=key_states.dtype))

        if seq_len < self.cache_max_size:
            self.cache_size = seq_len
            return past_key_values

        # 如果需要压缩的长度大于0，循环压缩所有完整chunk
        while (seq_len > self.recent_size and
               (seq_len - self.recent_size - self.last_compressed_idx) >= self.chunk_size):
            past_key_values = self.compress_chunk(
                key_states, value_states, layer_idx, past_key_values, seq_len, rotary_emb
            )

            seq_len = past_key_values.key_cache[layer_idx].shape[2]
            self.cache_size = seq_len

        return past_key_values
    
    def _update_importance_scores(self, attn_score_cache, num_key_value_groups):
        num_new_tokens = attn_score_cache.shape[2]
        attn_score_cache = sum_group(attn_score_cache, num_key_value_groups)
        attn_score_cache = attn_score_cache.sum(dim=0).sum(dim=1)

        if self.importance_scores is None:
            self.importance_scores = attn_score_cache
        else:
            attn_score_cache[:, :-num_new_tokens] += self.importance_scores
            self.importance_scores = attn_score_cache
    
    def compress_chunk(self, key_states, value_states, layer_idx, past_key_values, total_len, rotary_emb):
        start_idx = self.last_compressed_idx
        end_idx = min(start_idx + self.chunk_size, total_len - self.recent_size)

        if self.num_sum_tokens <= 0 and self.topk_important <= 0:
            new_key_cache = torch.cat([
                past_key_values.key_cache[layer_idx][:, :, :start_idx],
                past_key_values.key_cache[layer_idx][:, :, end_idx:]
            ], dim=2)
            
            new_value_cache = torch.cat([
                past_key_values.value_cache[layer_idx][:, :, :start_idx],
                past_key_values.value_cache[layer_idx][:, :, end_idx:]
            ], dim=2)
            
            past_key_values.key_cache[layer_idx] = new_key_cache
            past_key_values.value_cache[layer_idx] = new_value_cache

            self.importance_scores = torch.cat([
                self.importance_scores[:, :start_idx],
                self.importance_scores[:, end_idx:]
            ], dim=1)

            self.last_compressed_idx = start_idx

            return past_key_values   

        chunk_size = end_idx - start_idx

        key_chunk = past_key_values.key_cache[layer_idx][:, :, start_idx:end_idx]
        value_chunk = past_key_values.value_cache[layer_idx][:, :, start_idx:end_idx]
        scores_chunk = self.importance_scores[:, start_idx:end_idx]
        
        bsz, num_heads, chunk_len, head_dim = key_chunk.shape

        key_topk, value_topk, topk_scores, new_score_segment = None, None, None, None

        if self.topk_important > 0:
            # top-k important tokens
            _, topk_indices = torch.topk(scores_chunk, k=min(self.topk_important, chunk_len), dim=-1)
            topk_indices = topk_indices.sort(dim=-1).values
            topk_indices_exp = topk_indices.view(bsz, -1, self.topk_important, 1).expand(-1, -1, -1, head_dim)

            key_topk = torch.gather(key_chunk, dim=2, index=topk_indices_exp)
            value_topk = torch.gather(value_chunk, dim=2, index=topk_indices_exp)

            topk_scores = torch.gather(scores_chunk, 1, topk_indices)

        if self.num_sum_tokens > 0:
            # 可学习的 summary query 向量
            summary_q = self.summary_q.to(device=key_chunk.device)
            summary_q = summary_q.unsqueeze(0).expand(bsz, -1, -1, -1)  # [bsz, num_heads, num_sum_tokens, head_dim]

            chunk_attn_mask = torch.full(
                (self.num_sum_tokens, chunk_len + self.num_sum_tokens), 
                float('-inf'), 
                device=key_chunk.device,
                dtype=summary_q.dtype
            )

            chunk_attn_mask[:, :chunk_len] = 0

            for i in range(self.num_sum_tokens):
                # 允许关注之前的summary tokens
                chunk_attn_mask[i, chunk_len:chunk_len + i] = 0
                # 允许关注自己
                chunk_attn_mask[i, chunk_len + i] = 0
            
            chunk_attn_mask = chunk_attn_mask.unsqueeze(0).unsqueeze(0)  # [bsz, num_heads, num_sum_tokens, chunk_len + num_sum_tokens]
            
            extended_key = torch.cat([
                key_chunk, 
                torch.zeros(bsz, num_heads, self.num_sum_tokens, head_dim, device=key_chunk.device, dtype=summary_q.dtype)
            ], dim=2)
            
            extended_value = torch.cat([
                value_chunk, 
                torch.zeros(bsz, num_heads, self.num_sum_tokens, head_dim, device=key_chunk.device, dtype=summary_q.dtype)
            ], dim=2)

            # 迭代计算每个summary token
            summary_k_list = []
            summary_v_list = []
            summary_scores_list = []

            for i in range(self.num_sum_tokens):
                # 当前summary token的query
                cur_sum_q = summary_q[:, :, i:i+1, :]
                
                # 当前summary token的注意力分数
                cur_sum_attn_scores = torch.matmul(
                    cur_sum_q,
                    extended_key.transpose(-1, -2)
                ) / math.sqrt(head_dim)
                
                # 应用掩码
                cur_sum_attn_scores = cur_sum_attn_scores + chunk_attn_mask[:, :, i:i+1, :]
                
                cur_sum_attn_weights = F.softmax(cur_sum_attn_scores, dim=-1)
                
                # 聚合得到当前summary token的KV
                cur_sum_k = torch.matmul(cur_sum_attn_weights, extended_key)
                cur_sum_v = torch.matmul(cur_sum_attn_weights, extended_value)
                
                # 更新扩展的KV序列
                extended_key[:, :, chunk_len + i] = cur_sum_k.squeeze(2)
                extended_value[:, :, chunk_len + i] = cur_sum_v.squeeze(2)
                
                # 保存当前summary token的KV
                summary_k_list.append(cur_sum_k)
                summary_v_list.append(cur_sum_v)
                
                # 计算重要分数
                cur_sum_score = cur_sum_attn_scores.sum(dim=-1).sum(dim=0)
                summary_scores_list.append(cur_sum_score)
            
            # 合并所有summary tokens
            summary_k = torch.cat(summary_k_list, dim=2)
            summary_v = torch.cat(summary_v_list, dim=2)
            summary_scores = torch.cat(summary_scores_list, dim=1)





            # 计算 summary tokens 的 attention score
            # attn_scores = torch.matmul(             # [bsz, num_heads, num_sum_tokens, chunk_len]
            #     summary_q,
            #     key_chunk.transpose(-1, -2)
            # ) / math.sqrt(head_dim)

            # attn_weights = F.softmax(attn_scores, dim=-1)
            # L1归一化
            # attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

            # 使用注意力权重聚合 kv
            # summary_k = torch.matmul(attn_weights, key_chunk)
            # summary_v = torch.matmul(attn_weights, value_chunk)

            # summary tokens 的 important scores
            # summary_scores = attn_scores.sum(dim=-1).sum(dim=0)
            # summary_scores = attn_weights.mean(dim=[0, 3])

            # if topk_scores.numel() > 0:
            #     scale_factor = torch.log(topk_scores.mean() / (summary_scores.mean() + 1e-8))
            #     summary_scores = summary_scores * torch.exp(scale_factor.detach())

            if self.topk_important > 0:
                compressed_key = torch.cat([key_topk, summary_k], dim=2)
                compressed_value = torch.cat([value_topk, summary_v], dim=2)
                new_score_segment = torch.cat([topk_scores, summary_scores], dim=1)
            else:
                compressed_key = summary_k
                compressed_value = summary_v
                new_score_segment = summary_scores
        else:
            compressed_key = key_topk
            compressed_value = value_topk
            new_score_segment = topk_scores
        
        # 重新计算compressed chunk的RoPE
        # compressed_len = compressed_key.shape[2]
        # new_positions = torch.arange(
        #     start_idx, 
        #     start_idx + compressed_len,
        #     device=compressed_key.device
        # ).unsqueeze(0)  # [1, compressed_len]

        # cos, sin = rotary_emb(compressed_key, new_positions)
        # dummy_q = torch.zeros_like(compressed_key)
        # _, compressed_key = apply_rotary_pos_emb(
        #     dummy_q, compressed_key, cos, sin
        # )

        if past_key_values is None:
            past_key_values = DynamicCache()
            past_key_values.key_cache = [compressed_key]
            past_key_values.value_cache = [compressed_value]
            
            if chunk_len < key_states.shape[2]:
                remaining_key = key_states[:, :, chunk_size:]
                remaining_value = value_states[:, :, chunk_size:]
                past_key_values.update(remaining_key, remaining_value, layer_idx)
        else:
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

        self.importance_scores = torch.cat([
            self.importance_scores[:, :start_idx],
            new_score_segment,
            self.importance_scores[:, end_idx:]
        ], dim=1)

        self.last_compressed_idx += compressed_key.shape[2]
        
        return past_key_values
    
    def _clean_scores(self):
        self.importance_scores = None
        self.cache_size = 0