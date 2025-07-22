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
        sum_cache_size=512,
        recent_size=512,
        chunk_size=26,
        topk_important=4,
        num_sum_tokens=2,
        k_seq_dim=2,
        v_seq_dim=2,
        num_heads=32,
        head_dim=128,
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

        self.summary_q = nn.Parameter(torch.randn(num_heads, self.num_sum_tokens, head_dim))

    def __call__(self, key_states, value_states, layer_idx, past_key_values, attn_weights, rotary_emb, num_key_value_groups):
        """
        直接修改past_key_values，避免复制缓存
        """ 
        # 更新重要性分数：注意力权重列求和
        self._update_importance_scores(attn_weights, num_key_value_groups)

        seq_len = key_states.shape[2]
        if seq_len < self.cache_max_size:
            self.cache_size = seq_len
            return past_key_values

        # 如果需要压缩的长度大于0，则循环压缩所有完整chunk
        while (seq_len > self.recent_size and
               (seq_len - self.recent_size - self.last_compressed_idx) >= self.chunk_size):
            past_key_values = self.compress_chunk(
                key_states, value_states, layer_idx, past_key_values, seq_len, rotary_emb
            )

            # 更新缓存长度（压缩后长度会变化）
            seq_len = past_key_values.key_cache[layer_idx].shape[2]
            self.cache_size = seq_len

        return past_key_values
    
    def _update_importance_scores(self, attn_score_cache, num_key_value_groups):
        """更新token重要性分数（注意力权重列求和）"""
        # attn_weights形状: [bsz, num_heads, q_len, kv_len]
        num_new_tokens = attn_score_cache.shape[2]
        attn_score_cache = sum_group(attn_score_cache, num_key_value_groups)
        attn_score_cache = attn_score_cache.sum(dim=0).sum(dim=1)

        if self.importance_scores is None:
            self.importance_scores = attn_score_cache
        else:
            attn_score_cache[:, :-num_new_tokens] += self.importance_scores
            self.importance_scores = attn_score_cache
        
        # if self.importance_scores.shape[1] < num_new_tokens:
        #     # 扩展分数矩阵
        #     padding = num_new_tokens - self.importance_scores.shape[1]
        #     self.importance_scores = F.pad(self.importance_scores, (0, padding))
        
        # self.importance_scores[:, -num_new_tokens:] += attn_score_cache
    
    def compress_chunk(self, key_states, value_states, layer_idx, past_key_values, total_len, rotary_emb):
        # 计算要压缩的chunk范围
        start_idx = self.last_compressed_idx
        end_idx = min(start_idx + self.chunk_size, total_len - self.recent_size)
        chunk_size = end_idx - start_idx

        # 从 Cache 中获取chunk
        key_chunk = past_key_values.key_cache[layer_idx][:, :, start_idx:end_idx]
        value_chunk = past_key_values.value_cache[layer_idx][:, :, start_idx:end_idx]
        scores_chunk = self.importance_scores[:, start_idx:end_idx]
        
        bsz, _, chunk_len, head_dim = key_chunk.shape
        
        # top-k important tokens
        _, topk_indices = torch.topk(scores_chunk, k=min(self.topk_important, chunk_len), dim=-1)
        topk_indices = topk_indices.sort(dim=-1).values
        topk_indices_exp = topk_indices.view(bsz, -1, self.topk_important, 1).expand(-1, -1, -1, head_dim)

        key_topk = torch.gather(key_chunk, dim=2, index=topk_indices_exp)
        value_topk = torch.gather(value_chunk, dim=2, index=topk_indices_exp)

        topk_scores = torch.gather(scores_chunk, 1, topk_indices)

        if self.num_sum_tokens > 0:
            # 可学习的summary query向量
            summary_q = self.summary_q.to(device=key_chunk.device)
            summary_q = summary_q.unsqueeze(0).expand(bsz, -1, -1, -1)  # [bsz, num_heads, num_sum_tokens, head_dim]

            # 计算summary tokens的注意力分数
            attn_scores = torch.matmul(             # [bsz, num_heads, num_sum_tokens, chunk_len]
                summary_q,
                key_chunk.transpose(-1, -2)
            ) / math.sqrt(head_dim)

            # 应用softmax获取注意力权重
            attn_weights = F.softmax(attn_scores, dim=-1)
            # L1归一化
            # attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

            # 使用注意力权重聚合key
            summary_k = torch.matmul(attn_weights, key_chunk)
            # 使用注意力权重聚合value
            summary_v = torch.matmul(attn_weights, value_chunk)

            # summary tokens 的 important scores
            summary_scores = attn_scores.sum(dim=-1).sum(dim=0)
            # summary_scores = attn_weights.mean(dim=[0, 3])

            if topk_scores.numel() > 0:
                scale_factor = torch.log(topk_scores.mean() / (summary_scores.mean() + 1e-8))
                summary_scores = summary_scores * torch.exp(scale_factor.detach())

            # 组合重要token和summary tokens
            compressed_key = torch.cat([key_topk, summary_k], dim=2)
            compressed_value = torch.cat([value_topk, summary_v], dim=2)
        else:
            compressed_key = key_topk
            compressed_value = value_topk
        

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

        if self.num_sum_tokens > 0:
            new_score_segment = torch.cat([topk_scores, summary_scores], dim=1)
        else:
            new_score_segment = topk_scores

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