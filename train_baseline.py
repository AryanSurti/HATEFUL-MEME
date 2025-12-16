import os
import json
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import CLIPModel, CLIPProcessor, get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score


# -----------------------------
# Paths (match your structure)
# -----------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "archive", "data")
TRAIN_JSONL = os.path.join(DATA_DIR, "train.jsonl")
DEV_JSONL   = os.path.join(DATA_DIR, "dev.jsonl")
TEST_JSONL  = os.path.join(DATA_DIR, "test.jsonl")
IMG_DIR     = os.path.join(DATA_DIR, "img")

OUT_MODEL = os.path.join(ROOT, "best_model.pt")


# -----------------------------
# Reproducibility
# -----------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Utilities
# -----------------------------
def resolve_img_path(raw_img: str) -> str:
    p = (raw_img or "").replace("\\", "/").strip()
    if p.startswith("img/"):
        p = p[len("img/"):]
    return os.path.join(IMG_DIR, p)


# -----------------------------
# Dataset
# -----------------------------
class MemeDataset(Dataset):
    def __init__(self, jsonl_path: str, require_label: bool):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if require_label and "label" not in obj:
                    continue
                self.samples.append(obj)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = resolve_img_path(item.get("img", ""))
        text = item.get("text", "")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path} | raw={item.get('img')}")

        image = Image.open(img_path).convert("RGB")

        label = item.get("label", None)
        if label is None:
            label = -1
        return image, text, int(label)


# -----------------------------
# Model: CLIP features -> MLP
# -----------------------------
class ClipBinaryClassifier(nn.Module):
    def __init__(self, clip_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_name)
        self.hidden = self.clip.config.projection_dim  # usually 512

        # Slightly stronger head than before (still simple)
        self.head = nn.Sequential(
            nn.Linear(self.hidden * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        img_feat = self.clip.get_image_features(pixel_values=pixel_values)
        txt_feat = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

        img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-8)
        txt_feat = txt_feat / (txt_feat.norm(dim=-1, keepdim=True) + 1e-8)

        x = torch.cat([img_feat, txt_feat], dim=1)
        logits = self.head(x).squeeze(1)
        return logits


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def unfreeze_last_layers(clip_model: CLIPModel, n_vision: int = 2, n_text: int = 2):
    """
    Unfreeze only the LAST n layers of vision + text encoders + projection layers.
    This usually gives gains without destroying CLIP.
    """
    # Freeze everything first
    set_requires_grad(clip_model, False)

    # Unfreeze projection layers (important!)
    if hasattr(clip_model, "visual_projection"):
        set_requires_grad(clip_model.visual_projection, True)
    if hasattr(clip_model, "text_projection"):
        set_requires_grad(clip_model.text_projection, True)

    # Vision encoder last layers
    try:
        v_layers = clip_model.vision_model.encoder.layers
        for layer in v_layers[-n_vision:]:
            set_requires_grad(layer, True)
        set_requires_grad(clip_model.vision_model.post_layernorm, True)
    except Exception:
        pass

    # Text encoder last layers
    try:
        t_layers = clip_model.text_model.encoder.layers
        for layer in t_layers[-n_text:]:
            set_requires_grad(layer, True)
        set_requires_grad(clip_model.text_model.final_layer_norm, True)
    except Exception:
        pass


# -----------------------------
# Collate
# -----------------------------
@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    labels: torch.Tensor


def make_collate(processor: CLIPProcessor):
    def collate_fn(batch):
        images, texts, labels = zip(*batch)
        enc = processor(
            text=list(texts),
            images=list(images),
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        labels_t = torch.tensor(labels, dtype=torch.float32)
        return Batch(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            pixel_values=enc["pixel_values"],
            labels=labels_t
        )
    return collate_fn


# -----------------------------
# Eval
# -----------------------------
@torch.no_grad()
def eval_auc(model, loader, device):
    model.eval()
    ys = []
    ps = []
    for b in loader:
        logits = model(
            b.input_ids.to(device),
            b.attention_mask.to(device),
            b.pixel_values.to(device),
        )
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        ys.append(b.labels.numpy())
        ps.append(probs)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return float(roc_auc_score(y, p))


# -----------------------------
# Train
# -----------------------------
def train():
    seed_everything(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("ROOT:", ROOT)
    print("DATA_DIR:", DATA_DIR)
    print("IMG_DIR:", IMG_DIR)
    print("Device:", device)

    clip_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(clip_name)

    train_ds = MemeDataset(TRAIN_JSONL, require_label=True)
    dev_ds   = MemeDataset(DEV_JSONL, require_label=True)

    print("Train samples:", len(train_ds))
    print("Dev samples:", len(dev_ds))

    # Count imbalance for pos_weight
    labels = [lbl for (_, _, lbl) in train_ds]
    pos = sum(1 for x in labels if x == 1)
    neg = sum(1 for x in labels if x == 0)
    pos_weight = (neg / max(pos, 1.0))
    print(f"Train pos={pos} neg={neg} pos_weight={pos_weight:.4f}")

    # If you OOM, set batch_size=16
    batch_size = 32
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=make_collate(processor)
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=make_collate(processor)
    )

    model = ClipBinaryClassifier(clip_name=clip_name).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    # Mixed precision
    use_amp = (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # -----------------------------
    # Two-stage training
    # -----------------------------
    EPOCHS_HEAD = 3          # stage A: head only
    EPOCHS_FINETUNE = 7      # stage B: unfreeze last layers
    TOTAL_EPOCHS = EPOCHS_HEAD + EPOCHS_FINETUNE

    best_auc = -1.0

    # ---------- Stage A: head only ----------
    print("\n=== Stage A: Train HEAD only (CLIP frozen) ===")
    set_requires_grad(model.clip, False)
    set_requires_grad(model.head, True)

    opt_head = torch.optim.AdamW(model.head.parameters(), lr=2e-4, weight_decay=1e-3)
    steps_a = len(train_loader) * EPOCHS_HEAD
    sch_head = get_cosine_schedule_with_warmup(
        opt_head,
        num_warmup_steps=max(10, int(0.1 * steps_a)),
        num_training_steps=max(1, steps_a),
    )

    for ep in range(1, EPOCHS_HEAD + 1):
        model.train()
        total_loss = 0.0

        for b in tqdm(train_loader, desc=f"[A] Epoch {ep}/{EPOCHS_HEAD}"):
            opt_head.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(
                    b.input_ids.to(device),
                    b.attention_mask.to(device),
                    b.pixel_values.to(device),
                )
                loss = criterion(logits, b.labels.to(device))

            scaler.scale(loss).backward()
            scaler.unscale_(opt_head)
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
            scaler.step(opt_head)
            scaler.update()
            sch_head.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(len(train_loader), 1)
        dev_auc = eval_auc(model, dev_loader, device)
        print(f"\n[A] Epoch {ep}: train_loss={avg_loss:.4f} dev_AUROC={dev_auc:.4f}")

        if dev_auc > best_auc:
            best_auc = dev_auc
            torch.save(
                {
                    "model": model.state_dict(),
                    "clip_name": clip_name,
                    "stage": "head_only",
                    "pos_weight": float(pos_weight),
                    "epochs": TOTAL_EPOCHS,
                },
                OUT_MODEL
            )
            print(f"✅ Saved best model -> {OUT_MODEL}")

    # ---------- Stage B: partial CLIP finetune ----------
    print("\n=== Stage B: Fine-tune LAST CLIP layers + head ===")
    unfreeze_last_layers(model.clip, n_vision=2, n_text=2)
    set_requires_grad(model.head, True)

    # Param groups: head LR bigger, CLIP LR tiny
    clip_params = [p for p in model.clip.parameters() if p.requires_grad]
    head_params = [p for p in model.head.parameters() if p.requires_grad]

    opt_ft = torch.optim.AdamW(
        [
            {"params": clip_params, "lr": 1e-6, "weight_decay": 1e-2},
            {"params": head_params, "lr": 5e-5, "weight_decay": 1e-3},
        ]
    )

    steps_b = len(train_loader) * EPOCHS_FINETUNE
    sch_ft = get_cosine_schedule_with_warmup(
        opt_ft,
        num_warmup_steps=max(10, int(0.1 * steps_b)),
        num_training_steps=max(1, steps_b),
    )

    for ep in range(1, EPOCHS_FINETUNE + 1):
        model.train()
        total_loss = 0.0

        for b in tqdm(train_loader, desc=f"[B] Epoch {ep}/{EPOCHS_FINETUNE}"):
            opt_ft.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(
                    b.input_ids.to(device),
                    b.attention_mask.to(device),
                    b.pixel_values.to(device),
                )
                loss = criterion(logits, b.labels.to(device))

            scaler.scale(loss).backward()
            scaler.unscale_(opt_ft)
            torch.nn.utils.clip_grad_norm_(clip_params + head_params, 1.0)
            scaler.step(opt_ft)
            scaler.update()
            sch_ft.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(len(train_loader), 1)
        dev_auc = eval_auc(model, dev_loader, device)
        print(f"\n[B] Epoch {ep}: train_loss={avg_loss:.4f} dev_AUROC={dev_auc:.4f}")

        if dev_auc > best_auc:
            best_auc = dev_auc
            torch.save(
                {
                    "model": model.state_dict(),
                    "clip_name": clip_name,
                    "stage": "finetune_last_layers",
                    "pos_weight": float(pos_weight),
                    "epochs": TOTAL_EPOCHS,
                },
                OUT_MODEL
            )
            print(f"✅ Saved best model -> {OUT_MODEL}")

    print("\nDONE. Best Dev AUROC:", best_auc)


if __name__ == "__main__":
    train()
