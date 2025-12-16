"""
Web demo for Hateful Meme Detection using Graphormer + CLIP
Allows image upload and text input for easy testing.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import gradio as gr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.graphormer_model import GraphormerModel
from src.graph_builder import build_graph


class MemeClassifier:
    def __init__(self, checkpoint_path: str, device: torch.device):
        """Load the trained Graphormer model."""
        self.device = device
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Model not found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        cfg = checkpoint["config"]
        
        # Load CLIP
        clip_backbone = cfg.get("clip_backbone", "openai/clip-vit-base-patch32")
        self.clip_layer_idx = cfg.get("clip_layer_idx", -1)
        
        print(f"Loading CLIP: {clip_backbone}")
        self.processor = CLIPProcessor.from_pretrained(clip_backbone)
        self.clip_model = CLIPModel.from_pretrained(clip_backbone).to(device)
        
        if "clip_state" in checkpoint:
            self.clip_model.load_state_dict(checkpoint["clip_state"])
            print("✓ Loaded fine-tuned CLIP weights")
        
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False
        
        # Create projection layers
        shared_dim = cfg["input_dim"]
        text_dim = cfg.get("text_dim", shared_dim)
        vision_dim = cfg.get("vision_dim", 768)
        
        self.text_proj = nn.Identity() if text_dim == shared_dim else nn.Linear(text_dim, shared_dim)
        self.image_proj = nn.Linear(vision_dim, shared_dim)
        self.text_proj = self.text_proj.to(device)
        self.image_proj = self.image_proj.to(device)
        
        if "text_proj_state" in checkpoint:
            self.text_proj.load_state_dict(checkpoint["text_proj_state"])
        if "image_proj_state" in checkpoint:
            self.image_proj.load_state_dict(checkpoint["image_proj_state"])
        
        self.text_proj.eval()
        self.image_proj.eval()
        
        # Create Graphormer model
        self.model = GraphormerModel(
            input_dim=shared_dim,
            d_model=cfg["d_model"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            dropout=cfg.get("dropout", 0.1),
        ).to(device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        
        self.top_k = cfg.get("top_k", 4)
        
        print(f"✓ Model loaded successfully")
        print(f"  AUROC: {cfg.get('best_auc', 'N/A')}")
        print(f"  Architecture: {cfg['num_layers']} layers, {cfg['d_model']} dim, {self.top_k} top-K edges")
    
    def predict(self, image: Image.Image, text: str) -> dict:
        """Run inference on a single meme."""
        if image is None:
            return {
                "error": "Please upload an image",
                "prediction": "",
                "confidence": "",
                "explanation": ""
            }
        
        if not text or text.strip() == "":
            text = "[No text provided]"
        
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Process with CLIP
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs, output_hidden_states=True, return_dict=True)
            
            # Extract hidden states
            text_hidden = outputs.text_model_output.hidden_states[self.clip_layer_idx]
            vision_hidden = outputs.vision_model_output.hidden_states[self.clip_layer_idx]
            attention_mask = inputs["attention_mask"]
            
            # Build graph
            txt_len = int(attention_mask[0].sum().item())
            txt_start = 1
            txt_end = max(txt_len - 1, txt_start)
            text_feats = text_hidden[0, txt_start:txt_end]
            text_feats = self.text_proj(text_feats)
            
            image_feats = vision_hidden[0, 1:]  # Remove CLS
            image_feats = self.image_proj(image_feats)
            
            # Build graph
            graph = build_graph(
                text_feats=text_feats,
                image_feats=image_feats,
                top_k=self.top_k,
            )
            
            # Prepare batch tensors
            node_feats = graph["x"].unsqueeze(0)
            N = node_feats.size(1)
            node_mask = torch.ones(1, N, dtype=torch.bool, device=self.device)
            
            text_mask = torch.zeros(1, N, dtype=torch.bool, device=self.device)
            text_mask[0, graph["text_indices"]] = True
            
            image_mask = torch.zeros(1, N, dtype=torch.bool, device=self.device)
            image_mask[0, graph["image_indices"]] = True
            
            global_indices = graph["global_index"].unsqueeze(0)
            
            # Forward pass
            logits = self.model(
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
        
        # Format results
        if is_hateful:
            prediction_text = "🚫 HATEFUL CONTENT DETECTED"
            explanation = "This meme contains hateful or offensive content and should be moderated."
            color = "red"
        else:
            prediction_text = "✅ SAFE CONTENT"
            explanation = "This meme does not contain hateful content and is appropriate."
            color = "green"
        
        result_html = f"""
        <div style="padding: 20px; border-radius: 10px; background-color: {'#fee'; border: 2px solid #f00' if is_hateful else '#efe; border: 2px solid #0a0'};">
            <h2 style="color: {color}; margin-bottom: 10px;">{prediction_text}</h2>
            <p style="font-size: 18px; margin: 10px 0;">
                <strong>Confidence:</strong> {confidence:.1%}
            </p>
            <p style="font-size: 18px; margin: 10px 0;">
                <strong>Hateful Probability:</strong> {prob:.1%}
            </p>
            <p style="font-size: 16px; margin-top: 15px; color: #666;">
                {explanation}
            </p>
        </div>
        """
        
        return result_html


def create_demo(classifier: MemeClassifier):
    """Create Gradio interface."""
    
    def predict_wrapper(image, text):
        return classifier.predict(image, text)
    
    # Custom CSS for better appearance
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .gr-button-primary {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
    }
    """
    
    with gr.Blocks(css=custom_css, title="Hateful Meme Detector") as demo:
        gr.Markdown(
            """
            # 🔍 Hateful Meme Detection System
            ### Graph-based Multimodal Fusion with CLIP + Graphormer
            
            Upload a meme image and enter its text to detect hateful content using our research model.
            
            **Model Details:**
            - Architecture: Graphormer with CLIP Vision-Language Fusion
            - Training: Facebook Hateful Memes Dataset
            - Performance: 71.4% AUROC on validation set
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="📷 Upload Meme Image",
                    height=400
                )
                text_input = gr.Textbox(
                    label="📝 Enter Meme Text",
                    placeholder="Type the text from the meme here...",
                    lines=3
                )
                submit_btn = gr.Button("🔎 Analyze Meme", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                output = gr.HTML(label="📊 Prediction Result")
        
        # Examples
        gr.Markdown("### 📋 Try These Examples:")
        gr.Examples(
            examples=[
                ["archive/data/img/01235.png", "This is funny"],
                ["archive/data/img/01236.png", "Great meme"],
                ["archive/data/img/01243.png", "LOL"],
            ],
            inputs=[image_input, text_input],
            outputs=output,
            fn=predict_wrapper,
            cache_examples=False,
        )
        
        submit_btn.click(
            fn=predict_wrapper,
            inputs=[image_input, text_input],
            outputs=output
        )
        
        gr.Markdown(
            """
            ---
            ### 🎓 Research Information
            - **Novelty:** Graphormer-style graph fusion over CLIP token + patch nodes
            - **Edges:** Text-text chain, image-image grid, text-image similarity-weighted connections
            - **Training:** Fine-tuned last 2 CLIP layers with graph attention mechanism
            """
        )
    
    return demo


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="best_graphormer.pt")
    parser.add_argument("--share", action="store_true", help="Create public link (for 72h)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load model
    print("Loading model...")
    classifier = MemeClassifier(args.model, device)
    print("\n✓ Model ready!\n")
    
    # Create and launch demo
    demo = create_demo(classifier)
    
    print("="*60)
    print("🚀 Starting web demo...")
    print("="*60)
    
    if args.share:
        print("⚠️  Creating PUBLIC link (valid for 72 hours)")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()
