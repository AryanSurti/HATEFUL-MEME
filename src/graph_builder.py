import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

EDGE_TEXT_TEXT = 0
EDGE_IMG_IMG = 1
EDGE_TEXT_IMG = 2
EDGE_GLOBAL = 3


def _grid_neighbors(idx: int, grid: int) -> List[int]:
    row, col = divmod(idx, grid)
    neighbors = []
    if row > 0:
        neighbors.append((row - 1) * grid + col)
    if row < grid - 1:
        neighbors.append((row + 1) * grid + col)
    if col > 0:
        neighbors.append(row * grid + (col - 1))
    if col < grid - 1:
        neighbors.append(row * grid + (col + 1))
    return neighbors


def build_graph(
    text_feats: torch.Tensor,
    image_feats: torch.Tensor,
    top_k: int = 6,
) -> Dict[str, torch.Tensor]:
    """
    Build a heterogeneous graph from text/image embeddings.

    Args:
        text_feats: [T, d] text token embeddings (special tokens removed).
        image_feats: [P, d] patch embeddings (CLS removed).
        top_k: number of top patches per text token.
    """
    device = text_feats.device
    text_len, dim = text_feats.shape
    img_len = image_feats.shape[0]

    node_feats = torch.cat([text_feats, image_feats], dim=0)
    global_feat = node_feats.mean(dim=0, keepdim=True)
    x = torch.cat([node_feats, global_feat], dim=0)  # [N, d]

    edge_index: List[Tuple[int, int]] = []
    edge_type: List[int] = []
    edge_weight: List[float] = []

    # Text-text chain edges (bidirectional).
    for i in range(max(text_len - 1, 0)):
        edge_index.extend([(i, i + 1), (i + 1, i)])
        edge_type.extend([EDGE_TEXT_TEXT, EDGE_TEXT_TEXT])
        edge_weight.extend([0.0, 0.0])

    # Image-image grid edges (bidirectional 4-neighborhood).
    if img_len > 0:
        grid = int(math.sqrt(img_len))
        assert grid * grid == img_len, "Patch count must form a square grid."
        for p in range(img_len):
            neighbors = _grid_neighbors(p, grid)
            for n in neighbors:
                src = text_len + p
                dst = text_len + n
                edge_index.append((src, dst))
                edge_type.append(EDGE_IMG_IMG)
                edge_weight.append(0.0)

    # Text-image edges by cosine similarity between normalized embeddings.
    if text_len > 0 and img_len > 0:
        text_norm = F.normalize(text_feats, dim=-1, eps=1e-8)
        image_norm = F.normalize(image_feats, dim=-1, eps=1e-8)
        sim = torch.matmul(text_norm, image_norm.transpose(0, 1))  # [T, P]
        k = min(top_k, img_len)
        topk_vals, topk_idx = torch.topk(sim, k=k, dim=1)
        for t in range(text_len):
            for j in range(k):
                patch_idx = topk_idx[t, j].item()
                weight = topk_vals[t, j].item()
                t_node = t
                p_node = text_len + patch_idx
                # t -> p
                edge_index.append((t_node, p_node))
                edge_type.append(EDGE_TEXT_IMG)
                edge_weight.append(weight)
                # p -> t
                edge_index.append((p_node, t_node))
                edge_type.append(EDGE_TEXT_IMG)
                edge_weight.append(weight)

    # Global node connections (bidirectional).
    global_idx = text_len + img_len
    for node in range(global_idx):
        edge_index.extend([(global_idx, node), (node, global_idx)])
        edge_type.extend([EDGE_GLOBAL, EDGE_GLOBAL])
        edge_weight.extend([0.0, 0.0])

    edge_index_tensor = torch.tensor(edge_index, device=device, dtype=torch.long).t()
    edge_type_tensor = torch.tensor(edge_type, device=device, dtype=torch.long)
    edge_weight_tensor = torch.tensor(edge_weight, device=device, dtype=torch.float)

    text_indices = torch.arange(text_len, device=device, dtype=torch.long)
    image_indices = torch.arange(img_len, device=device, dtype=torch.long) + text_len

    return {
        "x": x.float(),
        "edge_index": edge_index_tensor,  # [2, E]
        "edge_type": edge_type_tensor,  # [E]
        "edge_weight": edge_weight_tensor,  # [E]
        "text_indices": text_indices,
        "image_indices": image_indices,
        "global_index": torch.tensor(global_idx, device=device, dtype=torch.long),
    }
