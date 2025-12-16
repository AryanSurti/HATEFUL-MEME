from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_builder import EDGE_GLOBAL, EDGE_IMG_IMG, EDGE_TEXT_IMG, EDGE_TEXT_TEXT


class GraphormerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert (
            d_model % num_heads == 0
        ), "d_model must be divisible by num_heads for multi-head attention."

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, d_model]
            attn_bias: [B, num_heads, N, N] additive bias
            node_mask: [B, N] boolean mask for valid nodes
        """
        B, N, _ = x.shape
        qkv = self.qkv(x)  # [B, N, 3*d]
        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim).permute(
            2, 0, 3, 1, 4
        )  # [3, B, H, N, D]
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)  # [B,H,N,N]
        scores = scores + attn_bias

        # Mask out invalid keys.
        key_mask = node_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,N]
        scores = scores.masked_fill(~key_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # [B,H,N,D]

        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        out = self.out_proj(out)
        out = self.dropout(out)
        x = self.norm1(x + out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class GraphormerModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.node_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList(
            [GraphormerLayer(d_model, num_heads, dropout) for _ in range(num_layers)]
        )
        self.edge_type_emb = nn.Embedding(4, num_heads)

        self.conflict_eps = 1e-6
        # MLP input: global + text_pool + image_pool + conflict + (optional external features)
        # We'll dynamically handle external features, default is d_model*3 + 1
        self.mlp_input_dim = d_model * 3 + 1
        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _build_attention_bias(
        self,
        edge_index_list: List[torch.Tensor],
        edge_type_list: List[torch.Tensor],
        edge_weight_list: List[torch.Tensor],
        max_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create attention bias tensor from edges."""
        bias = torch.zeros(
            len(edge_index_list), self.num_heads, max_nodes, max_nodes, device=device
        )
        et_emb = self.edge_type_emb.weight  # [4, num_heads]

        for b, (edge_index, edge_type, edge_weight) in enumerate(
            zip(edge_index_list, edge_type_list, edge_weight_list)
        ):
            if edge_index.numel() == 0:
                continue
            src = edge_index[0]
            dst = edge_index[1]
            et = edge_type
            ew = edge_weight
            bias[b, :, src, dst] += et_emb[et].transpose(0, 1)
            # Add similarity bias for text-img edges.
            text_img_mask = et == EDGE_TEXT_IMG
            if text_img_mask.any():
                sim_vals = ew[text_img_mask].unsqueeze(0)  # [1, E_textimg]
                bias[b, :, src[text_img_mask], dst[text_img_mask]] += sim_vals
        return bias

    def forward(
        self,
        node_feats: torch.Tensor,
        node_mask: torch.Tensor,
        text_mask: torch.Tensor,
        image_mask: torch.Tensor,
        global_indices: torch.Tensor,
        edge_index_list: List[torch.Tensor],
        edge_type_list: List[torch.Tensor],
        edge_weight_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            node_feats: [B, N, input_dim] padded node features.
            node_mask: [B, N] boolean mask for valid nodes.
            text_mask: [B, N] boolean mask for text nodes.
            image_mask: [B, N] boolean mask for image nodes.
            global_indices: [B] index of global node per graph.
            edge_index_list/type/weight: per-graph edge info (unbatched).
        Returns:
            logits: [B]
        """
        device = node_feats.device
        max_nodes = node_feats.size(1)

        attn_bias = self._build_attention_bias(
            edge_index_list, edge_type_list, edge_weight_list, max_nodes, device
        )

        x = self.node_proj(node_feats)
        for layer in self.layers:
            x = layer(x, attn_bias, node_mask)

        # Masks for pooling.
        text_mask_f = text_mask.float()
        image_mask_f = image_mask.float()
        text_count = text_mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        image_count = image_mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)

        text_pool = (x * text_mask_f.unsqueeze(-1)).sum(dim=1) / text_count
        image_pool = (x * image_mask_f.unsqueeze(-1)).sum(dim=1) / image_count

        batch_indices = torch.arange(x.size(0), device=device)
        global_embeds = x[batch_indices, global_indices]

        conflict = 1.0 - F.cosine_similarity(
            text_pool, image_pool, dim=-1, eps=self.conflict_eps
        )
        conflict = conflict.unsqueeze(-1)

        combined = torch.cat([global_embeds, text_pool, image_pool, conflict], dim=-1)
        logits = self.mlp(combined).squeeze(-1)
        return logits
