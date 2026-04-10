import torch
import torch.nn as nn


class CLIPElectraFusion(nn.Module):
    def __init__(
        self,
        clip_model,
        electra_model,
        fusion_img_dim=256,
        fusion_text_dim=256,
        num_classes=2,
        freeze_encoders=True,
        fusion_method="concatenate",
    ):
        super().__init__()
        self.clip = clip_model
        self.electra = electra_model
        self.num_classes = num_classes
        self.fusion_img_dim = fusion_img_dim
        self.fusion_text_dim = fusion_text_dim
        self.fusion_method = str(fusion_method).lower().strip()

        valid_fusion = {
            "concatenate",
            "addition",
            "multiplication",
            "gated_fusion",
            "attention_fusion",
            "bilinear_fusion",
        }
        if self.fusion_method not in valid_fusion:
            raise ValueError(
                f"Unknown fusion_method '{self.fusion_method}'. Supported: {sorted(valid_fusion)}"
            )

        if freeze_encoders:
            for p in self.clip.parameters():
                p.requires_grad = False
            for p in self.electra.parameters():
                p.requires_grad = False

        self.img_dim = clip_model.config.projection_dim
        electra_hidden_dim = electra_model.config.hidden_size

        if fusion_img_dim == self.img_dim:
            self.project_image = nn.Identity()
        else:
            self.project_image = nn.Sequential(
                nn.Linear(self.img_dim, fusion_img_dim),
                nn.GELU(),
                nn.LayerNorm(fusion_img_dim),
            )

        if fusion_text_dim == electra_hidden_dim:
            self.project_text = nn.Identity()
        else:
            self.project_text = nn.Sequential(
                nn.Linear(electra_hidden_dim, fusion_text_dim),
                nn.GELU(),
                nn.LayerNorm(fusion_text_dim),
            )

        self.same_modal_dim = fusion_img_dim == fusion_text_dim

        if self.fusion_method == "concatenate":
            self.fusion_dim = fusion_img_dim + fusion_text_dim
        elif self.fusion_method in {
            "addition",
            "multiplication",
            "gated_fusion",
            "attention_fusion",
        }:
            if not self.same_modal_dim:
                raise ValueError(
                    f"fusion_method='{self.fusion_method}' requires fusion_img_dim == fusion_text_dim, "
                    f"got {fusion_img_dim} vs {fusion_text_dim}."
                )
            self.fusion_dim = fusion_img_dim
        elif self.fusion_method == "bilinear_fusion":
            self.fusion_dim = fusion_img_dim + fusion_text_dim

        if self.fusion_method == "gated_fusion":
            self.gate_layer = nn.Linear(self.fusion_dim * 2, self.fusion_dim)

        if self.fusion_method == "attention_fusion":
            num_heads = 4 if (self.fusion_dim % 4 == 0) else 1
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=self.fusion_dim,
                num_heads=num_heads,
                batch_first=True,
            )

        if self.fusion_method == "bilinear_fusion":
            self.bilinear_fusion = nn.Bilinear(
                fusion_img_dim,
                fusion_text_dim,
                self.fusion_dim,
            )

        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.fusion_dim // 2, self.fusion_dim // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.fusion_dim // 4, num_classes),
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        if hasattr(self.clip, "get_image_features"):
            img_output = self.clip.get_image_features(pixel_values)
        else:
            img_output = self.clip(pixel_values=pixel_values)

        if hasattr(img_output, "image_embeds"):
            img_feats = img_output.image_embeds
        elif hasattr(img_output, "pooler_output"):
            img_feats = img_output.pooler_output
        elif isinstance(img_output, torch.Tensor):
            img_feats = img_output
        else:
            img_feats = img_output[0] if isinstance(img_output, (tuple, list)) else img_output

        img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-10)
        img_proj = self.project_image(img_feats)

        txt_out = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = txt_out.last_hidden_state

        attn = attention_mask.unsqueeze(-1).float()
        sum_emb = (last_hidden * attn).sum(dim=1)
        sum_mask = attn.sum(dim=1).clamp(min=1e-9)
        text_emb = sum_emb / sum_mask
        text_emb = text_emb / (text_emb.norm(dim=-1, keepdim=True) + 1e-10)

        text_proj = self.project_text(text_emb)

        if self.fusion_method == "concatenate":
            fused_rep = torch.cat([img_proj, text_proj], dim=-1)
        elif self.fusion_method == "addition":
            fused_rep = img_proj + text_proj
        elif self.fusion_method == "multiplication":
            fused_rep = img_proj * text_proj
        elif self.fusion_method == "gated_fusion":
            gate = torch.sigmoid(self.gate_layer(torch.cat([img_proj, text_proj], dim=-1)))
            fused_rep = gate * img_proj + (1.0 - gate) * text_proj
        elif self.fusion_method == "attention_fusion":
            tokens = torch.stack([img_proj, text_proj], dim=1)
            attn_out, _ = self.fusion_attention(tokens, tokens, tokens)
            fused_rep = attn_out.mean(dim=1)
        elif self.fusion_method == "bilinear_fusion":
            fused_rep = self.bilinear_fusion(img_proj, text_proj)
        else:
            raise RuntimeError(f"Unhandled fusion method: {self.fusion_method}")

        logits = self.classifier(fused_rep)
        return logits, img_proj, text_proj
