"""
Simple demo to test the Graphormer model on memes.
Upload an image and enter text to get hateful/not-hateful prediction.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.graphormer_model import GraphormerModel
from src.graph_builder import build_graph


def load_model(checkpoint_path: str, device: torch.device):
    """Load the trained Graphormer model."""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Model not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["config"]
    
    # Load CLIP
    clip_backbone = cfg.get("clip_backbone", "openai/clip-vit-base-patch32")
    clip_layer_idx = cfg.get("clip_layer_idx", -1)
    
    print(f"Loading CLIP: {clip_backbone}")
    processor = CLIPProcessor.from_pretrained(clip_backbone)
    clip_model = CLIPModel.from_pretrained(clip_backbone).to(device)
    
    if "clip_state" in checkpoint:
        clip_model.load_state_dict(checkpoint["clip_state"])
        print("✓ Loaded fine-tuned CLIP weights")
    
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    
    # Create projection layers
    shared_dim = cfg["input_dim"]
    text_dim = cfg.get("text_dim", shared_dim)
    vision_dim = cfg.get("vision_dim", 768)
    
    text_proj = nn.Identity() if text_dim == shared_dim else nn.Linear(text_dim, shared_dim)
    image_proj = nn.Linear(vision_dim, shared_dim)
    text_proj = text_proj.to(device)
    image_proj = image_proj.to(device)
    
    if "text_proj_state" in checkpoint:
        text_proj.load_state_dict(checkpoint["text_proj_state"])
    if "image_proj_state" in checkpoint:
        image_proj.load_state_dict(checkpoint["image_proj_state"])
    
    text_proj.eval()
    image_proj.eval()
    
    # Create Graphormer model
    model = GraphormerModel(
        input_dim=shared_dim,
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        dropout=cfg.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    top_k = cfg.get("top_k", 4)
    
    print(f"✓ Model loaded successfully")
    print(f"  Config: d_model={cfg['d_model']}, layers={cfg['num_layers']}, top_k={top_k}")
    
    return {
        "model": model,
        "clip_model": clip_model,
        "processor": processor,
        "text_proj": text_proj,
        "image_proj": image_proj,
        "top_k": top_k,
        "clip_layer_idx": clip_layer_idx,
        "device": device,
    }


def predict(image: Image.Image, text: str, model_dict: dict) -> dict:
    """Run inference on a single meme."""
    device = model_dict["device"]
    processor = model_dict["processor"]
    clip_model = model_dict["clip_model"]
    text_proj = model_dict["text_proj"]
    image_proj = model_dict["image_proj"]
    model = model_dict["model"]
    top_k = model_dict["top_k"]
    clip_layer_idx = model_dict["clip_layer_idx"]
    
    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Process with CLIP
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = clip_model(**inputs, output_hidden_states=True, return_dict=True)
        
        # Extract hidden states
        text_hidden = outputs.text_model_output.hidden_states[clip_layer_idx]  # [1, Lt, d]
        vision_hidden = outputs.vision_model_output.hidden_states[clip_layer_idx]  # [1, 1+P, d]
        attention_mask = inputs["attention_mask"]  # [1, Lt]
        
        # Build graph
        txt_len = int(attention_mask[0].sum().item())
        txt_start = 1
        txt_end = max(txt_len - 1, txt_start)
        text_feats = text_hidden[0, txt_start:txt_end]
        text_feats = text_proj(text_feats)
        
        image_feats = vision_hidden[0, 1:]  # Remove CLS
        image_feats = image_proj(image_feats)
        
        # Build graph
        graph = build_graph(
            text_feats=text_feats,
            image_feats=image_feats,
            top_k=top_k,
        )
        
        # Prepare batch tensors
        node_feats = graph["x"].unsqueeze(0)  # [1, N, d]
        N = node_feats.size(1)
        node_mask = torch.ones(1, N, dtype=torch.bool, device=device)
        
        text_mask = torch.zeros(1, N, dtype=torch.bool, device=device)
        text_mask[0, graph["text_indices"]] = True
        
        image_mask = torch.zeros(1, N, dtype=torch.bool, device=device)
        image_mask[0, graph["image_indices"]] = True
        
        global_indices = graph["global_index"].unsqueeze(0)  # [1]
        
        # Forward pass
        logits = model(
            node_feats,
            node_mask,
            text_mask,
            image_mask,
            global_indices,
            [graph["edge_index"]],
            [graph["edge_type"]],
            [graph["edge_weight"]],
        )
        
        prob = torch.sigmoid(logits).item()
    
    # Determine prediction
    is_hateful = prob > 0.5
    confidence = prob if is_hateful else (1 - prob)
    
    return {
        "prediction": "🚫 HATEFUL" if is_hateful else "✅ NOT HATEFUL",
        "probability_hateful": f"{prob:.1%}",
        "confidence": f"{confidence:.1%}",
        "verdict": "This meme contains hateful content" if is_hateful else "This meme is okay",
    }


def main():
    """Run interactive demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo for testing meme predictions")
    parser.add_argument("--model", type=str, default="best_graphormer.pt", help="Path to model checkpoint")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--text", type=str, help="Meme text")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading model...")
    model_dict = load_model(args.model, device)
    
    # If image and text provided, run single prediction
    if args.image and args.text:
        print(f"\n{'='*60}")
        print(f"Testing meme:")
        print(f"  Image: {args.image}")
        print(f"  Text: {args.text}")
        print(f"{'='*60}\n")
        
        image = Image.open(args.image).convert("RGB")
        result = predict(image, args.text, model_dict)
        
        print(f"Prediction: {result['prediction']}")
        print(f"Probability Hateful: {result['probability_hateful']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Verdict: {result['verdict']}")
        return
    
    # Otherwise, run interactive mode
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Enter image path and text to test predictions.")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            img_path = input("Image path (or 'quit'): ").strip()
            if img_path.lower() == 'quit':
                break
            
            if not Path(img_path).exists():
                print(f"❌ Image not found: {img_path}\n")
                continue
            
            text = input("Meme text: ").strip()
            
            print("\nProcessing...")
            image = Image.open(img_path).convert("RGB")
            result = predict(image, text, model_dict)
            
            print(f"\n{'-'*60}")
            print(f"📊 RESULT:")
            print(f"  {result['prediction']}")
            print(f"  Probability Hateful: {result['probability_hateful']}")
            print(f"  Confidence: {result['confidence']}")
            print(f"  {result['verdict']}")
            print(f"{'-'*60}\n")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    main()
