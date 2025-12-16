"""Simplified cross-attention fusion - more effective than graph for this task."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """Bidirectional cross-attention between text and image."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0
        
        # Text attends to image
        self.q_text = nn.Linear(dim, dim)
        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        
        # Image attends to text
        self.q_img = nn.Linear(dim, dim)
        self.k_text = nn.Linear(dim, dim)
        self.v_text = nn.Linear(dim, dim)
        
        self.out_text = nn.Linear(dim, dim)
        self.out_img = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text_feats, image_feats, text_mask=None):
        """
        Args:
            text_feats: [B, T, dim]
            image_feats: [B, P, dim]
            text_mask: [B, T] optional
        Returns:
            text_attended: [B, T, dim]
            image_attended: [B, P, dim]
        """
        B = text_feats.size(0)
        
        # Text → Image attention
        q_t = self.q_text(text_feats).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k_i = self.k_img(image_feats).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_i = self.v_img(image_feats).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_ti = torch.matmul(q_t, k_i.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_ti = torch.softmax(attn_ti, dim=-1)
        attn_ti = self.dropout(attn_ti)
        text_attended = torch.matmul(attn_ti, v_i)
        text_attended = text_attended.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.head_dim)
        text_attended = self.out_text(text_attended)
        
        # Image → Text attention
        q_i = self.q_img(image_feats).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k_t = self.k_text(text_feats).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_t = self.v_text(text_feats).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_it = torch.matmul(q_i, k_t.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if text_mask is not None:
            attn_it = attn_it.masked_fill(~text_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_it = torch.softmax(attn_it, dim=-1)
        attn_it = self.dropout(attn_it)
        image_attended = torch.matmul(attn_it, v_t)
        image_attended = image_attended.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.head_dim)
        image_attended = self.out_img(image_attended)
        
        return text_attended, image_attended


class ImprovedFusionModel(nn.Module):
    """Lightweight fusion with cross-attention and better regularization."""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        
        # Project to shared space
        self.text_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.image_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        
        # Cross-attention fusion
        self.cross_attn = CrossModalAttention(hidden_dim, num_heads=8, dropout=dropout)
        
        # Self-attention for refinement
        self.text_self_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
        self.img_self_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
        
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.img_norm = nn.LayerNorm(hidden_dim)
        
        # Classifier with heavy regularization
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 1, hidden_dim),  # concat text, img, max_pool, avg_pool, conflict
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )
        
    def forward(self, text_hidden, vision_hidden, attention_mask):
        """
        Args:
            text_hidden: [B, T, d_text]
            vision_hidden: [B, P, d_vision]
            attention_mask: [B, T]
        Returns:
            logits: [B]
        """
        B = text_hidden.size(0)
        
        # Remove special tokens from text (BOS/EOS)
        text_feats = text_hidden[:, 1:-1]  # [B, T-2, d]
        text_mask = attention_mask[:, 1:-1]  # [B, T-2]
        
        # Remove CLS from vision
        image_feats = vision_hidden[:, 1:]  # [B, P, d]
        
        # Project to shared space
        text_feats = self.text_proj(text_feats)
        image_feats = self.image_proj(image_feats)
        
        # Cross-modal attention
        text_cross, img_cross = self.cross_attn(text_feats, image_feats, text_mask)
        
        # Residual + self-attention
        text_feats = text_feats + text_cross
        image_feats = image_feats + img_cross
        
        text_self, _ = self.text_self_attn(text_feats, text_feats, text_feats, key_padding_mask=~text_mask)
        img_self, _ = self.img_self_attn(image_feats, image_feats, image_feats)
        
        text_feats = self.text_norm(text_feats + text_self)
        image_feats = self.img_norm(image_feats + img_self)
        
        # Multiple pooling strategies
        text_mask_expanded = text_mask.unsqueeze(-1).float()
        text_avg = (text_feats * text_mask_expanded).sum(dim=1) / text_mask_expanded.sum(dim=1).clamp(min=1)
        text_max, _ = text_feats.max(dim=1)
        
        img_avg = image_feats.mean(dim=1)
        img_max, _ = image_feats.max(dim=1)
        
        # Conflict signal
        conflict = 1.0 - F.cosine_similarity(text_avg, img_avg, dim=-1, eps=1e-8).unsqueeze(-1)
        
        # Concatenate all features
        combined = torch.cat([text_avg, img_avg, text_max, img_max, conflict], dim=-1)
        
        # Classify
        logits = self.classifier(combined).squeeze(-1)
        return logits
