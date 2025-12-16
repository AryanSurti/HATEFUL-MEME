"""Train improved cross-attention model with better regularization."""
import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from .dataset import MemeDataset
from .crossattention_model import ImprovedFusionModel


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


def extract_clip_features(
    processor: CLIPProcessor,
    model: CLIPModel,
    images: List,
    texts: List[str],
    device: torch.device,
    no_grad: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    text_hidden = outputs.text_model_output.hidden_states[-1]
    vision_hidden = outputs.vision_model_output.hidden_states[-1]
    attention_mask = inputs["attention_mask"]
    return text_hidden, vision_hidden, attention_mask


def evaluate(
    model: ImprovedFusionModel,
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    clip_model.eval()
    all_labels: List[float] = []
    all_probs: List[float] = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"]
            texts = batch["texts"]
            labels = batch["labels"].to(device)

            text_hidden, vision_hidden, attention_mask = extract_clip_features(
                processor, clip_model, images, texts, device, no_grad=True
            )
            logits = model(text_hidden, vision_hidden, attention_mask)
            probs = torch.sigmoid(logits)

            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5
    return auc


def compute_pos_weight(dataset: MemeDataset) -> torch.Tensor:
    labels = [int(s.get("label", 0)) for s in dataset.samples]
    pos = sum(labels)
    neg = len(labels) - pos
    pos_weight = neg / max(pos, 1)
    return torch.tensor(pos_weight, dtype=torch.float)


def unfreeze_clip_layers(clip_model: CLIPModel, n_layers: int = 3):
    """Unfreeze last n layers + projections of CLIP."""
    for p in clip_model.parameters():
        p.requires_grad = False
    
    # Projections
    if hasattr(clip_model, "visual_projection") and clip_model.visual_projection is not None:
        for p in clip_model.visual_projection.parameters():
            p.requires_grad = True
    if hasattr(clip_model, "text_projection") and clip_model.text_projection is not None:
        for p in clip_model.text_projection.parameters():
            p.requires_grad = True
    
    # Last n vision layers
    try:
        for layer in clip_model.vision_model.encoder.layers[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in clip_model.vision_model.post_layernorm.parameters():
            p.requires_grad = True
    except:
        pass
    
    # Last n text layers
    try:
        for layer in clip_model.text_model.encoder.layers[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in clip_model.text_model.final_layer_norm.parameters():
            p.requires_grad = True
    except:
        pass


def main():
    parser = argparse.ArgumentParser(description="Train improved fusion model.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--clip_lr", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    args = parser.parse_args()

    set_seed(42)
    base_dir = Path(__file__).resolve().parent.parent
    train_path = base_dir / "archive" / "data" / "train.jsonl"
    dev_path = base_dir / "archive" / "data" / "dev.jsonl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = MemeDataset(str(train_path), require_label=True)
    dev_ds = MemeDataset(str(dev_path), require_label=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    # Unfreeze last 3 layers
    unfreeze_clip_layers(clip_model, n_layers=3)
    clip_model.to(device)

    text_dim = clip_model.config.text_config.hidden_size
    vision_dim = clip_model.config.vision_config.hidden_size
    
    model = ImprovedFusionModel(
        input_dim=max(text_dim, vision_dim),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    pos_weight = compute_pos_weight(train_ds).to(device)
    # Use label smoothing via manual implementation
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Separate param groups
    clip_params = [p for p in clip_model.parameters() if p.requires_grad]
    fusion_params = list(model.parameters())
    
    optimizer = torch.optim.AdamW([
        {"params": clip_params, "lr": args.clip_lr, "weight_decay": args.weight_decay},
        {"params": fusion_params, "lr": args.lr, "weight_decay": args.weight_decay},
    ])
    
    # Cosine annealing with warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"Unfrozen CLIP params: {sum(p.numel() for p in clip_params):,}")
    print(f"Fusion params: {sum(p.numel() for p in fusion_params):,}")
    print(f"Training for {args.epochs} epochs with batch_size={args.batch_size}")

    best_auc = 0.0
    best_path = base_dir / "best_improved.pt"
    patience = 0
    max_patience = 4

    for epoch in range(1, args.epochs + 1):
        model.train()
        clip_model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            images = batch["images"]
            texts = batch["texts"]
            labels = batch["labels"].to(device)

            text_hidden, vision_hidden, attention_mask = extract_clip_features(
                processor, clip_model, images, texts, device, no_grad=False
            )
            logits = model(text_hidden, vision_hidden, attention_mask)
            
            # Label smoothing
            if args.label_smoothing > 0:
                labels_smooth = labels * (1 - args.label_smoothing) + 0.5 * args.label_smoothing
            else:
                labels_smooth = labels
            
            loss = criterion(logits, labels_smooth)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=0.5)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * labels.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = epoch_loss / len(train_ds)
        dev_auc = evaluate(model, clip_model, processor, dev_loader, device)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, dev_auc={dev_auc:.4f}")

        if dev_auc > best_auc:
            best_auc = dev_auc
            patience = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "clip_state": clip_model.state_dict(),
                    "config": {
                        "hidden_dim": args.hidden_dim,
                        "dropout": args.dropout,
                        "text_dim": text_dim,
                        "vision_dim": vision_dim,
                    },
                },
                best_path,
            )
            print(f"✓ New best model saved (dev_auc={dev_auc:.4f})")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\n{'='*50}")
    print(f"Best dev AUROC: {best_auc:.4f}")
    print(f"Model saved to: {best_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
