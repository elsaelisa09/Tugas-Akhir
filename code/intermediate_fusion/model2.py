"""
Model architecture WITHOUT Transformer fusion (Late Fusion approach).
This model concatenates CLIP and ELECTRA features and directly feeds to MLP classifier.
"""

import torch
import torch.nn as nn


class CLIPElectraFusionNoTransformer(nn.Module):
    """
    Multimodal fusion model WITHOUT Transformer layer.
    
    Architecture:
    1. CLIP Vision Encoder: Extracts image features (512-dim)
    2. ELECTRA Text Encoder: Extracts text features (768-dim)
    3. Text projection layer (768 -> 256)
    4. Simple concatenation (512 + 256 = 768)
    5. 3-layer MLP classifier
    
    Args:
        clip_model: Pre-trained CLIP model
        electra_model: Pre-trained ELECTRA model
        fusion_img_dim: Dimension for image features (default: 512)
        fusion_text_dim: Dimension for text projection (default: 256)
        num_classes: Number of output classes (default: 2)
        freeze_encoders: Whether to freeze CLIP and ELECTRA encoders
    """
    
    def __init__(self, clip_model, electra_model,
                 fusion_img_dim=512, fusion_text_dim=256,
                 num_classes=2, freeze_encoders=True):
        super().__init__()
        self.clip = clip_model
        self.electra = electra_model

        if freeze_encoders:
            for p in self.clip.parameters():
                p.requires_grad = False
            for p in self.electra.parameters():
                p.requires_grad = False

        # CLIP dimensi output image featurenya 512 (tidak perlu projection)
        self.img_dim = 512
        
        # Projection layer untuk text saja (768 -> 256)
        self.project_text = nn.Sequential(
            nn.Linear(768, fusion_text_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_text_dim)
        )

        # Fusion dimension: 512 (CLIP) + 256 (text projected) = 768
        self.fusion_dim = self.img_dim + fusion_text_dim

        # 3-layer MLP classifier (same as original)
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim // 2), # 768 -> 384
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.fusion_dim // 2, self.fusion_dim // 4), # 384 -> 192
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.fusion_dim // 4, num_classes) # 192 -> num_classes
        )

    def forward(self, pixel_values, input_ids, attention_mask):

        # Extract image features from CLIP
        img_feats = self.clip.get_image_features(pixel_values)
        # L2 normalization
        img_proj = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-10)

        # Extract text features from ELECTRA
        txt_out = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = txt_out.last_hidden_state
        # Shape: (batch_size, sequence_length, 768)

        # Mean pooling: rata-rata semua token (kecuali padding)
        attn = attention_mask.unsqueeze(-1).float()
        sum_emb = (last_hidden * attn).sum(dim=1)
        sum_mask = attn.sum(dim=1).clamp(min=1e-9)
        text_emb = sum_emb / sum_mask
        # Shape: (batch_size, 768)

        # Project text dari 768 -> 256 dimensi
        text_proj = self.project_text(text_emb)

        # LATE FUSION: Simple concatenation (NO Transformer!)
        # Langsung gabung image (512) + text (256) = 768
        fused_rep = torch.cat([img_proj, text_proj], dim=-1)
        # Shape: (batch_size, 768)

        # 3-layer MLP classifier
        logits = self.classifier(fused_rep)

        return logits, img_proj, text_proj


class EarlyStopping:

    def __init__(self, patience=3, mode='max'):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.num_bad = 0
        self.should_stop = False
        
    def step(self, value):

        if self.best is None:
            self.best = value
            self.num_bad = 0
            return True
            
        improve = (value > self.best) if self.mode == 'max' else (value < self.best)
        
        if improve:
            self.best = value
            self.num_bad = 0
            return True
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.should_stop = True
            return False
