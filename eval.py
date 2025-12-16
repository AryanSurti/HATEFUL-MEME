import os
import json
import csv
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import CLIPProcessor, CLIPModel


# ----------------------------
# Paths (match your structure)
# ----------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "archive", "data")
IMG_DIR = os.path.join(DATA_DIR, "img")

TRAIN_JSONL = os.path.join(DATA_DIR, "train.jsonl")
DEV_JSONL   = os.path.join(DATA_DIR, "dev.jsonl")
TEST_JSONL  = os.path.join(DATA_DIR, "test.jsonl")  # exists in your folder
CKPT_PATH   = os.path.join(ROOT, "best_model.pt")

OUT_PRED_CSV = os.path.join(ROOT, "test_predictions.csv")


# ----------------------------
# Data
# ----------------------------
class MemeDataset(Dataset):
    def __init__(self, jsonl_path: str, img_dir: str):
        self.samples: List[Dict[str, Any]] = []
        self.img_dir = img_dir

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def _get_img_rel(self, item: Dict[str, Any]) -> str:
        # handle common keys
        for k in ["img", "image", "img_path", "image_path"]:
            if k in item:
                return item[k]
        raise KeyError(f"No image key found in item keys={list(item.keys())}")

    def _get_text(self, item: Dict[str, Any]) -> str:
        return str(item.get("text", ""))

    def _get_label(self, item: Dict[str, Any]) -> int:
        # test may not have labels
        if "label" not in item:
            return -1
        return int(item["label"])

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        img_rel = self._get_img_rel(item)

        # fix common bad path like "img/img/xxxx.png"
        img_rel = img_rel.replace("\\", "/")
        while img_rel.startswith("./"):
            img_rel = img_rel[2:]
        img_rel = img_rel.replace("img/img/", "img/")

        img_path = os.path.join(self.img_dir, os.path.basename(img_rel)) \
            if "img/" in img_rel else os.path.join(self.img_dir, img_rel)

        # final fallback: if they gave "img/42953.png", join as-is under DATA_DIR
        if not os.path.exists(img_path):
            alt = os.path.join(DATA_DIR, img_rel)
            if os.path.exists(alt):
                img_path = alt

        image = Image.open(img_path).convert("RGB")
        text = self._get_text(item)
        y = self._get_label(item)

        # id for saving predictions (fallback to index)
        _id = item.get("id", idx)

        return _id, image, text, y


def collate_fn(batch):
    ids, images, texts, labels = zip(*batch)
    return list(ids), list(images), list(texts), torch.tensor(labels, dtype=torch.long)


# ----------------------------
# Model (matches checkpoint "head.*")
# ----------------------------
class ClipConcatHead(nn.Module):
    def __init__(self, clip_name: str, freeze_clip: bool = True):
        super().__init__()
        self.clip_name = clip_name
        self.clip = CLIPModel.from_pretrained(clip_name)
        self.processor = CLIPProcessor.from_pretrained(clip_name)

        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False

        # CLIP ViT-B/32 projection dim is 512 → concat image+text = 1024
        proj_dim = self.clip.config.projection_dim
        self.head = nn.Sequential(
            nn.Linear(proj_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, images: List[Image.Image], texts: List[str], device: torch.device):
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = self.clip(**inputs)
        img_emb = out.image_embeds
        txt_emb = out.text_embeds

        # normalize (stabilizes)
        img_emb = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-12)
        txt_emb = txt_emb / (txt_emb.norm(dim=-1, keepdim=True) + 1e-12)

        feat = torch.cat([img_emb, txt_emb], dim=-1)
        logit = self.head(feat).squeeze(-1)
        return logit


# ----------------------------
# Checkpoint loading (fixes your error)
# ----------------------------
def load_checkpoint_safe(path: str):
    ckpt = torch.load(path, map_location="cpu")

    # format A: {"state_dict": ..., "clip_name": ..., "freeze_clip": ...}
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
        return sd, ckpt.get("clip_name", "openai/clip-vit-base-patch32"), ckpt.get("freeze_clip", True)

    # format B: {"model": model.state_dict(), "clip_name": ...}
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
        return sd, ckpt.get("clip_name", "openai/clip-vit-base-patch32"), ckpt.get("freeze_clip", True)

    # format C: raw state_dict
    return ckpt, "openai/clip-vit-base-patch32", True


def remap_head_classifier_keys(state_dict: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    model_keys = set(model.state_dict().keys())
    sd_keys = set(state_dict.keys())

    # If checkpoint uses "classifier.*" but model uses "head.*"
    if any(k.startswith("classifier.") for k in sd_keys) and any(k.startswith("head.") for k in model_keys):
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith("classifier."):
                new_sd["head." + k[len("classifier."):]] = v
            else:
                new_sd[k] = v
        return new_sd

    # If checkpoint uses "head.*" but model uses "classifier.*"
    if any(k.startswith("head.") for k in sd_keys) and any(k.startswith("classifier.") for k in model_keys):
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith("head."):
                new_sd["classifier." + k[len("head."):]] = v
            else:
                new_sd[k] = v
        return new_sd

    return state_dict


# ----------------------------
# Eval
# ----------------------------
@torch.no_grad()
def eval_auc(model: ClipConcatHead, dl: DataLoader, device: torch.device) -> float:
    all_probs = []
    all_y = []

    for ids, images, texts, y in dl:
        y = y.to(device)
        mask = (y != -1)
        if mask.sum().item() == 0:
            continue

        logits = model(images, texts, device)
        probs = torch.sigmoid(logits)

        all_probs.append(probs[mask].detach().cpu())
        all_y.append(y[mask].detach().cpu())

    if not all_probs:
        return float("nan")

    all_probs = torch.cat(all_probs).numpy()
    all_y = torch.cat(all_y).numpy()
    return float(roc_auc_score(all_y, all_probs))


@torch.no_grad()
def write_test_predictions(model: ClipConcatHead, dl: DataLoader, device: torch.device, out_csv: str):
    ids_all = []
    probs_all = []

    for ids, images, texts, y in dl:
        logits = model(images, texts, device)
        probs = torch.sigmoid(logits).detach().cpu().tolist()
        ids_all.extend(ids)
        probs_all.extend(probs)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "prob_hateful"])
        for _id, p in zip(ids_all, probs_all):
            w.writerow([_id, float(p)])

    print(f"✅ Saved test predictions -> {out_csv} (n={len(ids_all)})")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("ROOT:", ROOT)
    print("DATA_DIR:", DATA_DIR)
    print("Device:", device)
    print("CKPT:", CKPT_PATH)

    state_dict, clip_name, freeze_clip = load_checkpoint_safe(CKPT_PATH)
    print("CLIP:", clip_name)
    print("freeze_clip:", freeze_clip)

    model = ClipConcatHead(clip_name=clip_name, freeze_clip=freeze_clip).to(device)

    # 🔥 Fix the exact thing your screenshot shows: head vs classifier
    state_dict = remap_head_classifier_keys(state_dict, model)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("⚠️ Missing keys:", missing)
    if unexpected:
        print("⚠️ Unexpected keys:", unexpected)

    dev_ds = MemeDataset(DEV_JSONL, IMG_DIR)
    test_ds = MemeDataset(TEST_JSONL, IMG_DIR)

    dev_dl = DataLoader(dev_ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn)

    dev_auc = eval_auc(model, dev_dl, device)
    print(f"✅ DEV AUROC: {dev_auc:.4f} (n={len(dev_ds)})")

    write_test_predictions(model, test_dl, device, OUT_PRED_CSV)


if __name__ == "__main__":
    main()
