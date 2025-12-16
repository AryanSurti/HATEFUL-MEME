"""Evaluate improved model and generate predictions."""
import csv
from pathlib import Path
from typing import List, Tuple

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from .dataset import MemeDataset
from .crossattention_model import ImprovedFusionModel


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
    text_hidden = outputs.text_model_output.hidden_states[-1]
    vision_hidden = outputs.vision_model_output.hidden_states[-1]
    attention_mask = inputs["attention_mask"]
    return text_hidden, vision_hidden, attention_mask


def run_dev_auc(
    model: ImprovedFusionModel,
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    clip_model.eval()
    labels_all: List[float] = []
    probs_all: List[float] = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"]
            texts = batch["texts"]
            labels = batch["labels"].to(device)

            text_hidden, vision_hidden, attention_mask = extract_clip_features(
                processor, clip_model, images, texts, device
            )
            logits = model(text_hidden, vision_hidden, attention_mask)
            probs = torch.sigmoid(logits)
            labels_all.extend(labels.cpu().tolist())
            probs_all.extend(probs.cpu().tolist())

    try:
        auc = roc_auc_score(labels_all, probs_all)
    except ValueError:
        auc = 0.5
    return auc


def predict_test(
    model: ImprovedFusionModel,
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    dataloader: DataLoader,
    device: torch.device,
) -> List[Tuple[str, float]]:
    model.eval()
    clip_model.eval()
    results: List[Tuple[str, float]] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting test", leave=False):
            images = batch["images"]
            texts = batch["texts"]
            ids = batch["ids"]

            text_hidden, vision_hidden, attention_mask = extract_clip_features(
                processor, clip_model, images, texts, device
            )
            logits = model(text_hidden, vision_hidden, attention_mask)
            probs = torch.sigmoid(logits).cpu().tolist()
            results.extend(list(zip(ids, probs)))
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(__file__).resolve().parent.parent
    best_path = base_dir / "best_improved.pt"
    
    if not best_path.exists():
        raise FileNotFoundError(
            f"Cannot find model at {best_path}. Train first with: python -m src.train_improved"
        )

    checkpoint = torch.load(best_path, map_location=device)
    cfg = checkpoint["config"]

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    
    if "clip_state" in checkpoint:
        clip_model.load_state_dict(checkpoint["clip_state"])
        print("✓ Loaded fine-tuned CLIP")
    
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    model = ImprovedFusionModel(
        input_dim=max(cfg.get("text_dim", 512), cfg.get("vision_dim", 768)),
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dev_ds = MemeDataset(str(base_dir / "archive" / "data" / "dev.jsonl"), require_label=True)
    test_ds = MemeDataset(str(base_dir / "archive" / "data" / "test.jsonl"), require_label=False)

    dev_loader = DataLoader(
        dev_ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_batch
    )
    test_loader = DataLoader(
        test_ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_batch
    )

    dev_auc = run_dev_auc(model, clip_model, processor, dev_loader, device)
    print(f"\n{'='*50}")
    print(f"Dev AUROC: {dev_auc:.4f}")
    print(f"{'='*50}\n")

    predictions = predict_test(model, clip_model, processor, test_loader, device)

    out_path = base_dir / "test_predictions_improved.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prob_hateful"])
        writer.writerows(predictions)

    print(f"✓ Saved test predictions to: {out_path}\n")


if __name__ == "__main__":
    main()
