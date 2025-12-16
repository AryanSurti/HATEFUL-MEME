import csv
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from .dataset import MemeDataset
from .graph_builder import build_graph
from .graphormer_model import GraphormerModel


def collate_batch(batch):
    images, texts, labels, ids = zip(*batch)
    labels_tensor = torch.tensor(labels, dtype=torch.float)
    return {
        "images": list(images),
        "texts": list(texts),
        "labels": labels_tensor,
        "ids": list(ids),
    }


def extract_clip_features(
    processor: CLIPProcessor,
    model: CLIPModel,
    images: List,
    texts: List[str],
    device: torch.device,
    layer_idx: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = processor(
        text=list(texts),
        images=list(images),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
    text_hidden = outputs.text_model_output.hidden_states[layer_idx]
    vision_hidden = outputs.vision_model_output.hidden_states[layer_idx]
    attention_mask = inputs["attention_mask"]
    return text_hidden, vision_hidden, attention_mask


def build_graphs_from_clip(
    text_hidden: torch.Tensor,
    vision_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    text_proj: torch.nn.Module,
    image_proj: torch.nn.Module,
    top_k: int,
) -> List[dict]:
    graphs = []
    B = text_hidden.size(0)
    for i in range(B):
        txt_len = int(attention_mask[i].sum().item())
        txt_start = 1
        txt_end = max(txt_len - 1, txt_start)
        text_feats = text_hidden[i, txt_start:txt_end]
        text_feats = text_proj(text_feats)
        image_feats = vision_hidden[i, 1:]
        image_feats = image_proj(image_feats)
        graphs.append(
            build_graph(
                text_feats=text_feats,
                image_feats=image_feats,
                top_k=top_k,
            )
        )
    return graphs


def prepare_batch_tensors(graphs: List[dict], device: torch.device):
    max_nodes = max(g["x"].size(0) for g in graphs)
    input_dim = graphs[0]["x"].size(1)
    batch_size = len(graphs)

    node_feats = torch.zeros(batch_size, max_nodes, input_dim, device=device)
    node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)
    text_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)
    image_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)
    global_indices = torch.zeros(batch_size, dtype=torch.long, device=device)

    edge_index_list, edge_type_list, edge_weight_list = [], [], []

    for i, g in enumerate(graphs):
        n = g["x"].size(0)
        node_feats[i, :n] = g["x"]
        node_mask[i, :n] = True
        text_mask[i, g["text_indices"]] = True
        image_mask[i, g["image_indices"]] = True
        global_indices[i] = g["global_index"]

        edge_index_list.append(g["edge_index"])
        edge_type_list.append(g["edge_type"])
        edge_weight_list.append(g["edge_weight"])

    return (
        node_feats,
        node_mask,
        text_mask,
        image_mask,
        global_indices,
        edge_index_list,
        edge_type_list,
        edge_weight_list,
    )


def run_dev_auc(
    model: GraphormerModel,
    text_proj: torch.nn.Module,
    image_proj: torch.nn.Module,
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    dataloader: DataLoader,
    device: torch.device,
    top_k: int,
    layer_idx: int = -1,
) -> float:
    model.eval()
    labels_all: List[float] = []
    probs_all: List[float] = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"]
            texts = batch["texts"]
            labels = batch["labels"].to(device)

            text_hidden, vision_hidden, attention_mask = extract_clip_features(
                processor, clip_model, images, texts, device, layer_idx=layer_idx
            )
            graphs = build_graphs_from_clip(
                text_hidden,
                vision_hidden,
                attention_mask,
                text_proj,
                image_proj,
                top_k,
            )
            (
                node_feats,
                node_mask,
                text_mask,
                image_mask,
                global_indices,
                edge_index_list,
                edge_type_list,
                edge_weight_list,
            ) = prepare_batch_tensors(graphs, device)

            logits = model(
                node_feats,
                node_mask,
                text_mask,
                image_mask,
                global_indices,
                edge_index_list,
                edge_type_list,
                edge_weight_list,
            )
            probs = torch.sigmoid(logits)
            labels_all.extend(labels.cpu().tolist())
            probs_all.extend(probs.cpu().tolist())

    try:
        auc = roc_auc_score(labels_all, probs_all)
    except ValueError:
        auc = 0.5
    return auc


def predict_test(
    model: GraphormerModel,
    text_proj: torch.nn.Module,
    image_proj: torch.nn.Module,
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    dataloader: DataLoader,
    device: torch.device,
    top_k: int,
    layer_idx: int = -1,
) -> List[Tuple[str, float]]:
    model.eval()
    results: List[Tuple[str, float]] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predict test", leave=False):
            images = batch["images"]
            texts = batch["texts"]
            ids = batch["ids"]

            text_hidden, vision_hidden, attention_mask = extract_clip_features(
                processor, clip_model, images, texts, device, layer_idx=layer_idx
            )
            graphs = build_graphs_from_clip(
                text_hidden,
                vision_hidden,
                attention_mask,
                text_proj,
                image_proj,
                top_k,
            )
            (
                node_feats,
                node_mask,
                text_mask,
                image_mask,
                global_indices,
                edge_index_list,
                edge_type_list,
                edge_weight_list,
            ) = prepare_batch_tensors(graphs, device)

            logits = model(
                node_feats,
                node_mask,
                text_mask,
                image_mask,
                global_indices,
                edge_index_list,
                edge_type_list,
                edge_weight_list,
            )
            probs = torch.sigmoid(logits).cpu().tolist()
            results.extend(list(zip(ids, probs)))
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(__file__).resolve().parent.parent
    best_path = base_dir / "best_graphormer.pt"
    if not best_path.exists():
        raise FileNotFoundError(
            f"Cannot find saved model at {best_path}. Train it first with python -m src.train_graphormer."
        )

    checkpoint = torch.load(best_path, map_location=device)
    cfg = checkpoint["config"]
    
    # Load backbone and layer index from config (fallback to defaults for old checkpoints)
    clip_backbone = cfg.get("clip_backbone", "openai/clip-vit-base-patch32")
    clip_layer_idx = cfg.get("clip_layer_idx", -1)
    
    print(f"\n{'='*50}")
    print(f"CLIP backbone: {clip_backbone}")
    print(f"Hidden state layer index: {clip_layer_idx}")
    print(f"{'='*50}\n")

    processor = CLIPProcessor.from_pretrained(clip_backbone)
    clip_model = CLIPModel.from_pretrained(clip_backbone).to(device)
    
    # Load CLIP state if it was fine-tuned
    if "clip_state" in checkpoint:
        clip_model.load_state_dict(checkpoint["clip_state"])
        print("✓ Loaded fine-tuned CLIP weights")
    
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    shared_dim = cfg["input_dim"]
    text_dim = cfg.get("text_dim", shared_dim)
    vision_dim = cfg.get("vision_dim", clip_model.config.vision_config.hidden_size)

    text_proj = nn.Identity() if text_dim == shared_dim else nn.Linear(text_dim, shared_dim)
    image_proj = nn.Linear(vision_dim, shared_dim)
    text_proj = text_proj.to(device)
    image_proj = image_proj.to(device)
    if "text_proj_state" in checkpoint:
        text_proj.load_state_dict(checkpoint["text_proj_state"])
    if "image_proj_state" in checkpoint:
        image_proj.load_state_dict(checkpoint["image_proj_state"])

    model = GraphormerModel(
        input_dim=shared_dim,
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    dev_ds = MemeDataset(str(base_dir / "archive" / "data" / "dev.jsonl"), require_label=True)
    test_ds = MemeDataset(str(base_dir / "archive" / "data" / "test.jsonl"), require_label=False)

    dev_loader = DataLoader(
        dev_ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_batch
    )
    test_loader = DataLoader(
        test_ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_batch
    )

    top_k = cfg.get("top_k", 4)
    dev_auc = run_dev_auc(
        model,
        text_proj,
        image_proj,
        clip_model,
        processor,
        dev_loader,
        device,
        top_k,
        layer_idx=clip_layer_idx,
    )
    print(f"Dev AUROC: {dev_auc:.4f}")

    predictions = predict_test(
        model,
        text_proj,
        image_proj,
        clip_model,
        processor,
        test_loader,
        device,
        top_k,
        layer_idx=clip_layer_idx,
    )

    out_path = base_dir / "test_predictions_graphormer.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prob_hateful"])
        writer.writerows(predictions)

    print(f"Saved test predictions to {out_path}")


if __name__ == "__main__":
    main()
