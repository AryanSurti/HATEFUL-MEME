from pathlib import Path
import json
from PIL import Image

# ----------------------------
# Paths
# ----------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "archive" / "data"

TRAIN_JSONL = DATA_DIR / "train.jsonl"
DEV_JSONL   = DATA_DIR / "dev.jsonl"
TEST_JSONL  = DATA_DIR / "test.jsonl"

def read_jsonl(path: Path, limit=None):
    items = []
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if limit is not None and len(items) >= limit:
                break
    return items

def count_labels(items):
    # Many test sets have no labels -> treat missing as None
    pos = sum(1 for x in items if x.get("label", None) == 1)
    neg = sum(1 for x in items if x.get("label", None) == 0)
    unk = sum(1 for x in items if "label" not in x)
    return len(items), pos, neg, unk

def resolve_image_path(data_dir: Path, img_field: str) -> Path:
    """
    JSONL usually stores: "img": "img/42953.png"
    Correct absolute path should be: archive/data/img/42953.png
    """
    p = Path(img_field)

    # Common case: already includes "img/...."
    cand1 = data_dir / p

    # If someone stores only filename: "42953.png"
    cand2 = data_dir / "img" / p

    # If path is weird, try just the filename inside data/img/
    cand3 = data_dir / "img" / p.name

    for c in (cand1, cand2, cand3):
        if c.exists():
            return c

    # If nothing exists, return the most likely and let caller print debugging
    return cand1

def main():
    print("ROOT:", ROOT)
    print("DATA_DIR:", DATA_DIR)

    train = read_jsonl(TRAIN_JSONL, limit=None)
    dev   = read_jsonl(DEV_JSONL,   limit=None)

    # test.jsonl may not exist or may be unlabeled; handle gently
    if TEST_JSONL.exists():
        test = read_jsonl(TEST_JSONL, limit=None)
    else:
        test = []

    ntr, ptr, ntr0, utr = count_labels(train)
    ndv, pdv, ndv0, udv = count_labels(dev)
    nts, pts, nts0, uts = count_labels(test)

    print(f"Train (n,pos,neg,unlabeled): ({ntr}, {ptr}, {ntr0}, {utr})")
    print(f"Dev   (n,pos,neg,unlabeled): ({ndv}, {pdv}, {ndv0}, {udv})")
    print(f"Test  (n,pos,neg,unlabeled): ({nts}, {pts}, {nts0}, {uts})")

    # Sample check
    sample = train[0]
    print("Sample keys:", list(sample.keys()))
    print("Sample:", sample)

    img_field = sample.get("img", "")
    if not img_field:
        raise ValueError("Sample has no 'img' field")

    img_path = resolve_image_path(DATA_DIR, img_field)

    print("Raw img field:", img_field)
    print("Resolved img path:", img_path)

    if not img_path.exists():
        # Print a helpful directory listing for debugging
        img_dir = DATA_DIR / "img"
        print("ERROR: Image file not found.")
        print("Expected:", img_path)
        print("DATA_DIR exists:", DATA_DIR.exists())
        print("IMG_DIR exists:", img_dir.exists())
        if img_dir.exists():
            some = sorted([p.name for p in img_dir.glob("*.png")])[:10]
            print("First 10 PNGs in data/img:", some)
        raise FileNotFoundError(img_path)

    img = Image.open(img_path).convert("RGB")
    print("Opened image OK. Size:", img.size, "Mode:", img.mode)

if __name__ == "__main__":
    main()
