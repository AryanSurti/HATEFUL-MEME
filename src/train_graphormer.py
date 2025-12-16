import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from .dataset import MemeDataset
from .graph_builder import build_graph
from .graphormer_model import GraphormerModel


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_batch(batch):
    images, texts, labels, ids = zip(*batch)
    labels_tensor = torch.tensor(labels, dtype=torch.float)
    return {
        "images": list(images),
        "texts": list(texts),
        "labels": labels_tensor,
        "ids": list(ids),
    }


def maybe_limit_dataset(dataset, limit):
    if limit is None:
        return dataset
    limit = min(limit, len(dataset))
    indices = list(range(limit))
    return Subset(dataset, indices)


def extract_clip_features(
    processor: CLIPProcessor,
    model: CLIPModel,
    images: List,
    texts: List[str],
    device: torch.device,
    layer_idx: int = -1,
    no_grad: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract per-token text and per-patch image hidden states from CLIP.
    
    Args:
        layer_idx: Which hidden state layer to use (-1 = last, -2 = second-to-last, etc.)
    """
    inputs = processor(
        text=list(texts),
        images=list(images),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if no_grad:
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
    else:
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
    
    # Use the same layer for both train and eval
    text_hidden = outputs.text_model_output.hidden_states[layer_idx]  # [B, Lt, d_text]
    vision_hidden = outputs.vision_model_output.hidden_states[layer_idx]  # [B, 1+P, d_vision]
    
    attention_mask = inputs["attention_mask"]  # [B, Lt]
    return text_hidden, vision_hidden, attention_mask


def build_graphs_from_clip(
    text_hidden: torch.Tensor,
    vision_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    text_proj: torch.nn.Module,
    image_proj: torch.nn.Module,
    top_k: int,
    debug_shapes: bool = False,
) -> List[dict]:
    graphs = []
    B = text_hidden.size(0)
    if debug_shapes:
        print(
            f"[DEBUG] text_hidden shape: {text_hidden.shape}, vision_hidden shape: {vision_hidden.shape}"
        )
    for i in range(B):
        txt_len = int(attention_mask[i].sum().item())
        txt_start = 1
        txt_end = max(txt_len - 1, txt_start)
        text_feats = text_hidden[i, txt_start:txt_end]
        expected_text_nodes = max(txt_len - 2, 0)
        assert text_feats.size(0) == expected_text_nodes, "Special tokens/padding handling mismatch"
        text_feats = text_proj(text_feats)

        image_feats = vision_hidden[i, 1:]
        assert image_feats.size(0) == vision_hidden.size(1) - 1, "CLS token not removed properly"
        image_feats = image_proj(image_feats)

        if debug_shapes:
            print(
                f"[DEBUG] sample {i}: text_nodes={text_feats.size(0)} "
                f"(expected {expected_text_nodes}, attention_tokens={txt_len}), "
                f"image_nodes={image_feats.size(0)}"
            )

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


def evaluate(
    model: GraphormerModel,
    text_proj: torch.nn.Module,
    image_proj: torch.nn.Module,
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    dataloader: DataLoader,
    device: torch.device,
    top_k: int,
    layer_idx: int = -1,
    debug_shapes: bool = False,
) -> float:
    model.eval()
    text_proj.eval()
    image_proj.eval()
    clip_model.eval()
    all_labels: List[float] = []
    all_probs: List[float] = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"]
            texts = batch["texts"]
            labels = batch["labels"].to(device)

            text_hidden, vision_hidden, attention_mask = extract_clip_features(
                processor, clip_model, images, texts, device, layer_idx=layer_idx, no_grad=True
            )
            graphs = build_graphs_from_clip(
                text_hidden,
                vision_hidden,
                attention_mask,
                text_proj,
                image_proj,
                top_k,
                debug_shapes,
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

            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # Handle edge case where only one class is present.
        auc = 0.5
    return auc


def compute_pos_weight(dataset: MemeDataset) -> torch.Tensor:
    labels = [int(s.get("label", 0)) for s in dataset.samples]
    pos = sum(labels)
    neg = len(labels) - pos
    pos_weight = neg / max(pos, 1)
    return torch.tensor(pos_weight, dtype=torch.float)


def main():
    parser = argparse.ArgumentParser(description="Train Graphormer fusion model.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip_lr", type=float, default=5e-6)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--unfreeze_clip", action="store_true", default=True)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_dev", type=int, default=None)
    parser.add_argument("--debug_shapes", action="store_true", default=False)
    parser.add_argument("--overfit_200", action="store_true", default=False)
    parser.add_argument("--clip_backbone", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--clip_layer_idx", type=int, default=-1, help="Hidden state layer index (-1=last, -2=second-to-last)")
    args = parser.parse_args()

    set_seed(42)
    base_dir = Path(__file__).resolve().parent.parent
    train_path = base_dir / "archive" / "data" / "train.jsonl"
    dev_path = base_dir / "archive" / "data" / "dev.jsonl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = MemeDataset(str(train_path), require_label=True)
    dev_ds = MemeDataset(str(dev_path), require_label=True)

    train_limit = args.limit_train
    dev_limit = args.limit_dev
    if args.overfit_200:
        train_limit = 200
        dev_limit = 200
        print("Overfit mode: limiting train/dev to 200 samples.")

    limited_train_ds = maybe_limit_dataset(train_ds, train_limit)
    limited_dev_ds = maybe_limit_dataset(dev_ds, dev_limit)
    print(
        f"Train samples: {len(limited_train_ds)} (limit={train_limit}), "
        f"Dev samples: {len(limited_dev_ds)} (limit={dev_limit})"
    )

    train_loader = DataLoader(
        limited_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
    )
    dev_loader = DataLoader(
        limited_dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    print(f"\n{'='*50}")
    print(f"CLIP backbone: {args.clip_backbone}")
    print(f"Hidden state layer index: {args.clip_layer_idx}")
    print(f"{'='*50}\n")
    
    processor = CLIPProcessor.from_pretrained(args.clip_backbone)
    clip_model = CLIPModel.from_pretrained(args.clip_backbone)
    
    # Unfreeze last layers of CLIP for better performance
    if args.unfreeze_clip:
        # Freeze everything first
        for p in clip_model.parameters():
            p.requires_grad = False
        
        # Unfreeze projections
        if hasattr(clip_model, "visual_projection") and clip_model.visual_projection is not None:
            for p in clip_model.visual_projection.parameters():
                p.requires_grad = True
        if hasattr(clip_model, "text_projection") and clip_model.text_projection is not None:
            for p in clip_model.text_projection.parameters():
                p.requires_grad = True
        
        # Unfreeze last 2 vision layers
        try:
            for layer in clip_model.vision_model.encoder.layers[-2:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in clip_model.vision_model.post_layernorm.parameters():
                p.requires_grad = True
        except:
            pass
        
        # Unfreeze last 2 text layers
        try:
            for layer in clip_model.text_model.encoder.layers[-2:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in clip_model.text_model.final_layer_norm.parameters():
                p.requires_grad = True
        except:
            pass
    else:
        for p in clip_model.parameters():
            p.requires_grad = False
    
    clip_model.to(device)

    def _count_params(modules):
        return sum(p.numel() for module in modules for p in module.parameters() if p.requires_grad)

    if args.unfreeze_clip:
        try:
            text_blocks = clip_model.text_model.encoder.layers[-2:]
            vision_blocks = clip_model.vision_model.encoder.layers[-2:]
            text_params = _count_params(text_blocks)
            vision_params = _count_params(vision_blocks)
        except AttributeError:
            text_params = 0
            vision_params = 0

        proj_params = 0
        for attr in ("text_projection", "visual_projection"):
            proj_module = getattr(clip_model, attr, None)
            if proj_module is not None:
                if isinstance(proj_module, nn.Module):
                    proj_params += sum(
                        p.numel() for p in proj_module.parameters() if p.requires_grad
                    )
                else:
                    proj_params += proj_module.numel() if proj_module.requires_grad else 0
        clip_trainable = sum(
            p.numel() for p in clip_model.parameters() if p.requires_grad
        )
        print(f"Trainable CLIP text params (last 2 layers): {text_params:,}")
        print(f"Trainable CLIP vision params (last 2 layers): {vision_params:,}")
        print(f"Trainable CLIP projection params: {proj_params:,}")
        print(f"Total trainable CLIP params: {clip_trainable:,}")
    else:
        print("CLIP frozen: no trainable CLIP parameters.")

    vision_cfg = clip_model.vision_model.config
    grid_size = vision_cfg.image_size // vision_cfg.patch_size

    text_dim = clip_model.config.text_config.hidden_size
    vision_dim = clip_model.config.vision_config.hidden_size
    shared_dim = text_dim  # project vision to text hidden size

    text_proj = nn.Identity() if text_dim == shared_dim else nn.Linear(text_dim, shared_dim)
    image_proj = nn.Linear(vision_dim, shared_dim)
    text_proj = text_proj.to(device)
    image_proj = image_proj.to(device)

    model = GraphormerModel(
        input_dim=shared_dim,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    
    print(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    pos_weight = compute_pos_weight(train_ds).to(device)
    base_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Label smoothing for binary classification
    def criterion(logits, labels):
        if args.label_smoothing > 0:
            smoothed = labels * (1 - args.label_smoothing) + args.label_smoothing * 0.5
            return base_criterion(logits, smoothed)
        return base_criterion(logits, labels)
    
    # Separate learning rates for CLIP and Graphormer
    clip_params = [p for p in clip_model.parameters() if p.requires_grad]
    graphormer_params = list(model.parameters()) + list(text_proj.parameters()) + list(image_proj.parameters())
    
    if clip_params:
        optimizer = torch.optim.AdamW([
            {"params": clip_params, "lr": args.clip_lr, "weight_decay": args.weight_decay * 0.1},
            {"params": graphormer_params, "lr": args.lr, "weight_decay": args.weight_decay},
        ])
        print(f"Training with unfrozen CLIP (lr={args.clip_lr}) and Graphormer (lr={args.lr})")
    else:
        optimizer = torch.optim.AdamW(graphormer_params, lr=args.lr, weight_decay=args.weight_decay)
        print(f"Training with frozen CLIP and Graphormer (lr={args.lr})")

    for idx, group in enumerate(optimizer.param_groups):
        params_count = sum(p.numel() for p in group["params"])
        print(
            f"Param group {idx}: lr={group['lr']:.1e}, "
            f"weight_decay={group.get('weight_decay', 0):.2e}, params={params_count:,}"
        )
    
    # ReduceLROnPlateau: reduces LR when dev AUROC plateaus
    # This helps prevent overfitting after the initial peak
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',  # maximize AUROC
        factor=0.5,  # reduce LR by half
        patience=2,  # wait 2 epochs without improvement
        verbose=True,
        min_lr=1e-7
    )
    print(f"Using ReduceLROnPlateau scheduler (factor=0.5, patience=2)")

    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_path = base_dir / "best_graphormer.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        text_proj.train()
        image_proj.train()
        clip_model.train()  # Important: set to train mode if we're updating params
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            images = batch["images"]
            texts = batch["texts"]
            labels = batch["labels"].to(device)

            text_hidden, vision_hidden, attention_mask = extract_clip_features(
                processor, clip_model, images, texts, device, layer_idx=args.clip_layer_idx
            )
            graphs = build_graphs_from_clip(
                text_hidden,
                vision_hidden,
                attention_mask,
                text_proj,
                image_proj,
                args.top_k,
                args.debug_shapes,
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
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(text_proj.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(image_proj.parameters(), max_norm=1.0)
            if clip_params:
                torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)

        train_loss = epoch_loss / len(train_ds)
        dev_auc = evaluate(
            model,
            text_proj,
            image_proj,
            clip_model,
            processor,
            dev_loader,
            device,
            top_k=args.top_k,
            layer_idx=args.clip_layer_idx,
            debug_shapes=args.debug_shapes,
        )

        # Step scheduler based on dev AUROC (reduces LR if plateau detected)
        scheduler.step(dev_auc)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, dev_auc={dev_auc:.4f}, lr={current_lr:.2e}")

        if dev_auc > best_auc:
            best_auc = dev_auc
            best_epoch = epoch
            patience_counter = 0
            checkpoint = {
                "model_state": model.state_dict(),
                "text_proj_state": text_proj.state_dict(),
                "image_proj_state": image_proj.state_dict(),
                "config": {
                    "input_dim": shared_dim,
                    "text_dim": text_dim,
                    "vision_dim": vision_dim,
                    "d_model": args.d_model,
                    "num_heads": args.num_heads,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "top_k": args.top_k,
                    "grid_size": grid_size,
                    "unfreeze_clip": args.unfreeze_clip,
                    "clip_backbone": args.clip_backbone,
                    "clip_layer_idx": args.clip_layer_idx,
                },
            }
            if args.unfreeze_clip and clip_params:
                checkpoint["clip_state"] = clip_model.state_dict()
            torch.save(checkpoint, best_path)
            print(f"✓ New best model saved (dev_auc={dev_auc:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epoch(s)")
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (best was epoch {best_epoch})")
                break

    print(f"\n{'='*50}")
    print(f"Best dev AUROC: {best_auc:.4f} (epoch {best_epoch})")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
