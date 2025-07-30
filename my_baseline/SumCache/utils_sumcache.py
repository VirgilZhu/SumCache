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


class TrioKVCache_LayerWise:
    def __init__(
        self,
        compress_cache_limit=4096-32,
        recent_size=32,
        chunk_size=64,
        topk_important=1,
        num_sum_tokens=2,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        self.compress_cache_limit = compress_cache_limit
        self.recent_size = recent_size
        self.cache_max_size = compress_cache_limit + recent_size
        self.chunk_size = chunk_size
        self.cache_size = 0
        self.topk_important = topk_important
        self.num_sum_tokens = num_sum_tokens
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim

        self.sum_compress_ratio = 0

        self.important_cache_limit = 0
        self.summary_cache_limit = 0
        
        self.important_cache_size = 0
        self.summary_cache_size = 0
        
        self.importance_scores = None
        self.last_compressed_idx = 0

        self.summary_q = None

    def __call__(self, key_states, value_states, layer_idx, past_key_values, attn_weights, rotary_emb, num_key_value_groups):
        self._update_importance_scores(attn_weights, num_key_value_groups)

        _, num_heads, seq_len, head_dim = key_states.shape

        # if self.summary_q is None:
        #     self.summary_q = nn.Parameter(
        #         torch.zeros(key_states.shape[0], num_heads, self.num_sum_tokens, head_dim, dtype=key_states.dtype, device=key_states.device))

        if seq_len < self.cache_max_size:
            self.cache_size = seq_len
            return past_key_values

        # 如果需要压缩的长度大于0，循环压缩所有完整chunk
        while (seq_len > self.recent_size and
               (seq_len - self.recent_size - self.last_compressed_idx) >= self.chunk_size):
            past_key_values = self.compress_chunk(
                layer_idx, past_key_values, rotary_emb
            )

            seq_len = past_key_values.key_cache[layer_idx].shape[2]
            self.cache_size = seq_len

        # 压缩summary_cache如果已满
        if self.summary_cache_size > self.summary_cache_limit:
            past_key_values = self.compress_summary_cache(
                layer_idx, past_key_values
            )
            seq_len = past_key_values.key_cache[layer_idx].shape[2]
            self.cache_size = seq_len
        
        # 压缩important_cache如果已满
        if self.important_cache_size > self.important_cache_limit:
            past_key_values = self.compress_important_cache(
                layer_idx, past_key_values
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

    
    def compress_chunk(self, layer_idx, past_key_values, rotary_emb):
        start_idx = self.last_compressed_idx
        end_idx = start_idx + self.chunk_size

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

        key_chunk = past_key_values.key_cache[layer_idx][:, :, start_idx:end_idx]
        value_chunk = past_key_values.value_cache[layer_idx][:, :, start_idx:end_idx]
        scores_chunk = self.importance_scores[:, start_idx:end_idx]
        
        bsz, num_heads, chunk_len, head_dim = key_chunk.shape

        # key_topk, value_topk, topk_scores, new_score_segment = None, None, None, None
        key_topk, value_topk, topk_scores = None, None, None
        summary_k, summary_v, summary_scores = None, None, None
        
        if self.topk_important > 0:
            _, topk_indices = torch.topk(scores_chunk, k=min(self.topk_important, chunk_len), dim=-1)
            topk_indices = topk_indices.sort(dim=-1).values
            topk_indices_exp = topk_indices.view(bsz, -1, self.topk_important, 1).expand(-1, -1, -1, head_dim)

            key_topk = torch.gather(key_chunk, dim=2, index=topk_indices_exp)
            value_topk = torch.gather(value_chunk, dim=2, index=topk_indices_exp)

            topk_scores = torch.gather(scores_chunk, 1, topk_indices)

        if self.num_sum_tokens > 0:
            summary_q = self.summary_q.expand(bsz, -1, -1, -1)  # [bsz, num_heads, num_sum_tokens, head_dim]
            
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
            
            chunk_attn_mask = chunk_attn_mask.unsqueeze(0).unsqueeze(0).expand(-1, num_heads, -1, -1)  # [bsz, num_heads, num_sum_tokens, chunk_len + num_sum_tokens]
            
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
                valid_scores = cur_sum_attn_scores[:, :, :, :-(self.num_sum_tokens - i - 1)]
                if i == self.num_sum_tokens - 1:
                    valid_scores = cur_sum_attn_scores
                cur_sum_score = valid_scores.sum(dim=-1).sum(dim=0)
                summary_scores_list.append(cur_sum_score)
            
            # 合并所有summary tokens
            summary_k = torch.cat(summary_k_list, dim=2)
            summary_v = torch.cat(summary_v_list, dim=2)
            summary_scores = torch.cat(summary_scores_list, dim=1)

            ### ———————————————————————————————————————————————————————————————————————————————————————————— ###

            # summary_q = self.summary_q

            # # 计算 summary tokens 的 attention score
            # attn_scores = torch.matmul(             # [bsz, num_heads, num_sum_tokens, chunk_len]
            #     summary_q,
            #     key_chunk.transpose(-1, -2)
            # ) / math.sqrt(head_dim)

            # attn_weights = F.softmax(attn_scores, dim=-1)

            # summary_k = torch.matmul(attn_weights, key_chunk)
            # summary_v = torch.matmul(attn_weights, value_chunk)

            # summary_scores = attn_scores.sum(dim=-1).sum(dim=0)
            # summary_scores = attn_weights.mean(dim=[0, 3])

            # if topk_scores.numel() > 0:
            #     scale_factor = torch.log(topk_scores.mean() / (summary_scores.mean() + 1e-8))
            #     summary_scores = summary_scores * torch.exp(scale_factor.detach())

            if self.topk_important > 0:
                # compressed_key = torch.cat([key_topk, summary_k], dim=2)
                # compressed_value = torch.cat([value_topk, summary_v], dim=2)
                # new_score_segment = torch.cat([topk_scores, summary_scores], dim=1)
                new_key_cache = torch.cat([
                    past_key_values.key_cache[layer_idx][:, :, :self.summary_cache_size],
                    summary_k,
                    past_key_values.key_cache[layer_idx][:, :, self.summary_cache_size:self.summary_cache_size+self.important_cache_size],
                    key_topk,
                    past_key_values.key_cache[layer_idx][:, :, self.summary_cache_size+self.important_cache_size:start_idx],
                    past_key_values.key_cache[layer_idx][:, :, end_idx:]
                ], dim=2)
                
                new_value_cache = torch.cat([
                    past_key_values.value_cache[layer_idx][:, :, :self.summary_cache_size],
                    summary_v,
                    past_key_values.value_cache[layer_idx][:, :, self.summary_cache_size:self.summary_cache_size+self.important_cache_size],
                    value_topk,
                    past_key_values.value_cache[layer_idx][:, :, self.summary_cache_size+self.important_cache_size:start_idx],
                    past_key_values.value_cache[layer_idx][:, :, end_idx:]
                ], dim=2)

                self.importance_scores = torch.cat([
                    self.importance_scores[:, :self.summary_cache_size],
                    summary_scores,
                    self.importance_scores[:, self.summary_cache_size:self.summary_cache_size+self.important_cache_size],
                    topk_scores,
                    self.importance_scores[:, self.summary_cache_size+self.important_cache_size:start_idx],
                    self.importance_scores[:, end_idx:]
                ], dim=1)
            else:
                # compressed_key = summary_k
                # compressed_value = summary_v
                # new_score_segment = summary_scores
                new_key_cache = torch.cat([
                    past_key_values.key_cache[layer_idx][:, :, :self.summary_cache_size],
                    summary_k,
                    past_key_values.key_cache[layer_idx][:, :, self.summary_cache_size:start_idx],
                    past_key_values.key_cache[layer_idx][:, :, end_idx:]
                ], dim=2)
                
                new_value_cache = torch.cat([
                    past_key_values.value_cache[layer_idx][:, :, :self.summary_cache_size],
                    summary_v,
                    past_key_values.value_cache[layer_idx][:, :, self.summary_cache_size:start_idx],
                    past_key_values.value_cache[layer_idx][:, :, end_idx:]
                ], dim=2)

                self.importance_scores = torch.cat([
                    self.importance_scores[:, :self.summary_cache_size],
                    summary_scores,
                    self.importance_scores[:, self.summary_cache_size:start_idx],
                    self.importance_scores[:, end_idx:]
                ], dim=1)
        else:
            # compressed_key = key_topk
            # compressed_value = value_topk
            # new_score_segment = topk_scores
            new_key_cache = torch.cat([
                past_key_values.key_cache[layer_idx][:, :, :self.summary_cache_size+self.important_cache_size],
                key_topk,
                past_key_values.key_cache[layer_idx][:, :, self.summary_cache_size+self.important_cache_size:start_idx],
                past_key_values.key_cache[layer_idx][:, :, end_idx:]
            ], dim=2)
            
            new_value_cache = torch.cat([
                past_key_values.value_cache[layer_idx][:, :, :self.summary_cache_size+self.important_cache_size],
                value_topk,
                past_key_values.value_cache[layer_idx][:, :, self.summary_cache_size+self.important_cache_size:start_idx],
                past_key_values.value_cache[layer_idx][:, :, end_idx:]
            ], dim=2)

            self.importance_scores = torch.cat([
                self.importance_scores[:, :self.summary_cache_size+self.important_cache_size],
                topk_scores,
                self.importance_scores[:, self.summary_cache_size+self.important_cache_size:start_idx],
                self.importance_scores[:, end_idx:]
            ], dim=1)
            
        past_key_values.key_cache[layer_idx] = new_key_cache
        past_key_values.value_cache[layer_idx] = new_value_cache
        
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

        # new_key_cache = torch.cat([
        #     past_key_values.key_cache[layer_idx][:, :, :start_idx],
        #     compressed_key,
        #     past_key_values.key_cache[layer_idx][:, :, end_idx:]
        # ], dim=2)
        
        # new_value_cache = torch.cat([
        #     past_key_values.value_cache[layer_idx][:, :, :start_idx],
        #     compressed_value,
        #     past_key_values.value_cache[layer_idx][:, :, end_idx:]
        # ], dim=2)
        
        # past_key_values.key_cache[layer_idx] = new_key_cache
        # past_key_values.value_cache[layer_idx] = new_value_cache

        # self.importance_scores = torch.cat([
        #     self.importance_scores[:, :start_idx],
        #     new_score_segment,
        #     self.importance_scores[:, end_idx:]
        # ], dim=1)

        self.last_compressed_idx += (key_topk.shape[2] + summary_k.shape[2])
        
        if key_topk is not None:
            self.important_cache_size += key_topk.shape[2]
        if summary_k is not None:
            self.summary_cache_size += summary_k.shape[2]
        
        return past_key_values 
    
    def compress_important_cache(self, layer_idx, past_key_values):
        start_idx = self.summary_cache_size
        end_idx = self.summary_cache_size + self.important_cache_size

        important_key = past_key_values.key_cache[layer_idx][:, :, start_idx:end_idx]
        important_value = past_key_values.value_cache[layer_idx][:, :, start_idx:end_idx]
        important_scores = self.importance_scores[:, start_idx:end_idx]
        
        bsz, _, _, head_dim = important_key.shape
        
        _, topk_indices = torch.topk(important_scores, k=self.important_cache_limit, dim=-1)
        topk_indices = topk_indices.sort(dim=-1).values
        topk_indices_exp = topk_indices.view(bsz, -1, self.important_cache_limit, 1).expand(-1, -1, -1, head_dim)

        keep_key = torch.gather(important_key, dim=2, index=topk_indices_exp)
        keep_value = torch.gather(important_value, dim=2, index=topk_indices_exp)
        keep_scores = torch.gather(important_scores, 1, topk_indices)
        
        new_key_cache = torch.cat([
            past_key_values.key_cache[layer_idx][:, :, :start_idx],
            keep_key,
            past_key_values.key_cache[layer_idx][:, :, end_idx:]
        ], dim=2)
        
        new_value_cache = torch.cat([
            past_key_values.value_cache[layer_idx][:, :, :start_idx],
            keep_value,
            past_key_values.value_cache[layer_idx][:, :, end_idx:]
        ], dim=2)
        
        new_scores = torch.cat([
            self.importance_scores[:, :start_idx],
            keep_scores,
            self.importance_scores[:, end_idx:]
        ], dim=1)
        
        past_key_values.key_cache[layer_idx] = new_key_cache
        past_key_values.value_cache[layer_idx] = new_value_cache

        self.importance_scores = new_scores
        self.important_cache_size = self.important_cache_limit
        self.last_compressed_idx -= (end_idx - start_idx - self.important_cache_size)
        
        return past_key_values

    def compress_summary_cache(self, layer_idx, past_key_values):
        start_idx = 0
        end_idx = self.summary_cache_size

        if end_idx <= 1:
            return past_key_values

        summary_key = past_key_values.key_cache[layer_idx][:, :, start_idx:end_idx]
        summary_value = past_key_values.value_cache[layer_idx][:, :, start_idx:end_idx]
        summary_scores = self.importance_scores[:, start_idx:end_idx]

        bsz, num_heads, num_tokens, head_dim = summary_key.shape

        remain = max(1, self.summary_cache_size // 2)

        key_A = summary_key[:, :, 0::2]
        key_B = summary_key[:, :, 1::2]
        value_A = summary_value[:, :, 0::2]
        value_B = summary_value[:, :, 1::2]

        num_A = key_A.shape[2]
        num_B = key_B.shape[2]

        # 计算A到B的相似度矩阵
        key_A_flat = key_A.permute(0, 2, 1, 3).flatten(2)  # [bsz, num_A, num_heads*head_dim]
        key_B_flat = key_B.permute(0, 2, 1, 3).flatten(2)  # [bsz, num_B, num_heads*head_dim]

        # 计算余弦相似度
        similarity_matrix = torch.cosine_similarity(
            key_A_flat.unsqueeze(2),  # [bsz, num_A, 1, dim]
            key_B_flat.unsqueeze(1),  # [bsz, 1, num_B, dim]
            dim=-1
        ).squeeze(0)  # [num_A, num_B]

        edges = []
        for i in range(num_A):
            # 找到最相似的B token
            max_similarity, best_j = torch.max(similarity_matrix[i], dim=0)
            edges.append((i, best_j.item(), max_similarity.item()))
        
        # 按相似度排序并保留前remain条边
        edges.sort(key=lambda x: x[2], reverse=True)
        selected_edges = edges[:remain]

        # 构建合并映射：记录每个B token被哪些A token连接
        b_to_a_map = {}  # b_idx -> list of (a_idx, similarity)
        selected_edge_set = set((i, j) for i, j, _ in selected_edges)

        for i, j, sim in selected_edges:
            if j not in b_to_a_map:
                b_to_a_map[j] = []
            b_to_a_map[j].append((i, sim))

        # 构建新的token序列
        new_keys = []
        new_values = []
        new_scores = []

        # 处理所有被连接的B tokens（合并组）
        processed_a = set()
        for j in range(num_B):
            if j in b_to_a_map:
                # 这个B token有连接，需要合并
                connected_a_tokens = b_to_a_map[j]
                
                # 收集所有需要合并的tokens（A tokens + 这个B token）
                merge_keys = []
                merge_values = []
                merge_scores_list = []
                
                # 添加所有连接的A tokens
                for a_idx, _ in connected_a_tokens:
                    merge_keys.append(key_A[:, :, a_idx])
                    merge_values.append(value_A[:, :, a_idx])
                    merge_scores_list.append(summary_scores[:, a_idx * 2])  # 原始索引
                    processed_a.add(a_idx)
                
                # 添加B token
                merge_keys.append(key_B[:, :, j])
                merge_values.append(value_B[:, :, j])
                merge_scores_list.append(summary_scores[:, j * 2 + 1])  # 原始索引
                
                # 计算平均值进行合并
                merged_key = torch.stack(merge_keys).mean(dim=0)
                merged_value = torch.stack(merge_values).mean(dim=0)
                merged_score = torch.stack(merge_scores_list, dim=1).mean(dim=1)
                
                new_keys.append(merged_key)
                new_values.append(merged_value)
                new_scores.append(merged_score)
        
        # 处理未被连接的A tokens
        for i in range(num_A):
            if i not in processed_a:
                new_keys.append(key_A[:, :, i])
                new_values.append(value_A[:, :, i])
                new_scores.append(summary_scores[:, i * 2])
        
        # 处理未被连接的B tokens
        for j in range(num_B):
            if j not in b_to_a_map:
                new_keys.append(key_B[:, :, j])
                new_values.append(value_B[:, :, j])
                new_scores.append(summary_scores[:, j * 2 + 1])
        
        new_keys = torch.cat(new_keys, dim=0).unsqueeze(0).expand(bsz, -1, -1, -1).transpose(1, 2)
        new_values = torch.cat(new_values, dim=0).unsqueeze(0).expand(bsz, -1, -1, -1).transpose(1, 2)
        new_scores = torch.stack(new_scores, dim=1).squeeze(0)

        # 重构整个KV cache
        new_key_cache = torch.cat([
            past_key_values.key_cache[layer_idx][:, :, :start_idx],
            new_keys,
            past_key_values.key_cache[layer_idx][:, :, end_idx:]
        ], dim=2)
        
        new_value_cache = torch.cat([
            past_key_values.value_cache[layer_idx][:, :, :start_idx],
            new_values,
            past_key_values.value_cache[layer_idx][:, :, end_idx:]
        ], dim=2)
        
        # 更新importance_scores
        new_importance_scores = torch.cat([
            self.importance_scores[:, :start_idx],
            new_scores,
            self.importance_scores[:, end_idx:]
        ], dim=1)
        
        # 更新缓存
        past_key_values.key_cache[layer_idx] = new_key_cache
        past_key_values.value_cache[layer_idx] = new_value_cache

        self.importance_scores = new_importance_scores
        self.summary_cache_size = new_keys.shape[2]
        self.last_compressed_idx -= (end_idx - start_idx - self.summary_cache_size)
        
        return past_key_values
    
    def _clean_scores(self):
        self.importance_scores = None
        self.cache_size = 0
        self.important_cache_size = 0
        self.summary_cache_size = 0
