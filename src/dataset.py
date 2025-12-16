import json
from pathlib import Path
from typing import Tuple

from PIL import Image
from torch.utils.data import Dataset


class MemeDataset(Dataset):
    """Dataset for meme detection with robust path handling."""

    def __init__(self, jsonl_path: str, require_label: bool = True) -> None:
        self.jsonl_path = Path(jsonl_path)
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Missing jsonl file: {self.jsonl_path}")
        self.require_label = require_label
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f]

        self.img_root = self.jsonl_path.parent / "img"

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_path(self, img_field: str) -> Path:
        # Normalize separators and strip leading "img/" if present.
        normalized = img_field.replace("\\", "/")
        if normalized.startswith("img/"):
            normalized = normalized[len("img/") :]
        return self.img_root / normalized

    def __getitem__(self, idx: int) -> Tuple[Image.Image, str, int, str]:
        sample = self.samples[idx]
        sample_id = str(sample.get("id", idx))
        text = sample.get("text", "")
        img_field = sample.get("img", "")
        label_val = sample.get("label", None)

        if self.require_label:
            if label_val is None:
                raise ValueError(f"Label missing for sample {sample_id}")
            label = int(label_val)
        else:
            label = -1

        img_path = self._resolve_image_path(img_field)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        return image, text, label, sample_id
