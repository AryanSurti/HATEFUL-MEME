# Hateful Meme Detection using Graph-based Multimodal Fusion

Research project implementing Graphormer-style graph neural networks over CLIP embeddings for hateful meme detection.

## 🎯 Model Architecture

- **Backbone**: CLIP (ViT-L/14) for vision-language features
- **Graph Construction**: Heterogeneous graph with text tokens, image patches, and global node
- **Fusion**: Graphormer transformer with edge-type attention bias
- **Performance**: 71.4% AUROC on validation set

## 📊 Key Innovation

Graphormer-style attention over CLIP features:
- **Nodes**: Text tokens, image patches, global aggregation node
- **Edges**: 
  - Text-text sequential connections
  - Image-image spatial grid connections  
  - Text-image similarity-weighted connections (top-K)
  - Global node connections to all nodes

## 🚀 Quick Start

### Training

```bash
python -m src.train_graphormer
```

### Evaluation

```bash
python -m src.eval_graphormer
```

### Demo (Google Colab)

Upload `Meme_Detection_Demo.ipynb` to Google Colab for interactive testing.

## 📁 Project Structure

```
├── src/
│   ├── dataset.py          # Data loading
│   ├── graph_builder.py    # Graph construction
│   ├── graphormer_model.py # Model architecture
│   ├── train_graphormer.py # Training script
│   └── eval_graphormer.py  # Evaluation script
├── demo.py                 # CLI demo
├── web_demo.py            # Gradio web interface
└── Meme_Detection_Demo.ipynb  # Google Colab notebook
```

## 📦 Requirements

```bash
pip install torch torchvision transformers pillow numpy scikit-learn tqdm gradio
```

## 🎓 Citation

Dataset: Facebook Hateful Memes Challenge  
https://ai.facebook.com/tools/hatefulmemes/
