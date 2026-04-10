# Potongan Kode Penting Model Multimodal

Berikut adalah empat potongan kode utama yang paling relevan untuk menjelaskan konfigurasi model multimodal pada penelitian ini, yaitu pembekuan encoder dan projection layer, proses ekstraksi fitur dan normalisasi, simple fusion, serta transformer fusion.

## 1. Freeze Encoder dan Projection Layer

Potongan kode ini menunjukkan bahwa encoder utama CLIP dan ELECTRA dibekukan selama pelatihan, sedangkan fitur gambar dan teks diproyeksikan ke dimensi fusion tertentu sebelum digabungkan.

```python
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
        nn.LayerNorm(fusion_img_dim)
    )

if fusion_text_dim == electra_hidden_dim:
    self.project_text = nn.Identity()
else:
    self.project_text = nn.Sequential(
        nn.Linear(electra_hidden_dim, fusion_text_dim),
        nn.GELU(),
        nn.LayerNorm(fusion_text_dim)
    )
```

## 2. Extraction, Pooling, dan Normalization

Potongan kode berikut menunjukkan proses ekstraksi fitur visual dari CLIP, ekstraksi fitur tekstual dari ELECTRA, masked mean pooling pada modalitas teks, serta normalisasi L2 pada kedua modalitas sebelum diproyeksikan.

```python
if hasattr(self.clip, 'get_image_features'):
    img_output = self.clip.get_image_features(pixel_values)
else:
    img_output = self.clip(pixel_values=pixel_values)

if hasattr(img_output, 'image_embeds'):
    img_feats = img_output.image_embeds
elif hasattr(img_output, 'pooler_output'):
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
```

## 3. Simple Fusion Block

Potongan kode ini menunjukkan beberapa metode fusion sederhana yang digunakan dalam penelitian, yaitu `concatenate`, `addition`, `multiplication`, `gated_fusion`, `attention_fusion`, dan `bilinear_fusion`.

```python
if self.fusion_method == 'concatenate':
    fused_rep = torch.cat([img_proj, text_proj], dim=-1)
elif self.fusion_method == 'addition':
    fused_rep = img_proj + text_proj
elif self.fusion_method == 'multiplication':
    fused_rep = img_proj * text_proj
elif self.fusion_method == 'gated_fusion':
    gate = torch.sigmoid(self.gate_layer(torch.cat([img_proj, text_proj], dim=-1)))
    fused_rep = gate * img_proj + (1.0 - gate) * text_proj
elif self.fusion_method == 'attention_fusion':
    tokens = torch.stack([img_proj, text_proj], dim=1)
    attn_out, _ = self.fusion_attention(tokens, tokens, tokens)
    fused_rep = attn_out.mean(dim=1)
elif self.fusion_method == 'bilinear_fusion':
    fused_rep = self.bilinear_fusion(img_proj, text_proj)

logits = self.classifier(fused_rep)
```

## 4. Transformer Fusion Block

Potongan kode berikut menunjukkan implementasi fusion berbasis transformer. Representasi teks dan gambar dibentuk sebagai token multimodal, ditambahkan positional embedding, lalu diproses menggunakan Transformer Encoder dua lapis dengan `nhead=8` sebelum diteruskan ke classifier.

```python
self.pos_embedding = nn.Parameter(torch.randn(1, 2, self.fusion_dim))

encoder_layer = nn.TransformerEncoderLayer(
    d_model=self.fusion_dim,
    nhead=8,
    dim_feedforward=self.fusion_dim * 4,
    dropout=0.1,
    batch_first=True
)
self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
```

```python
text_token = torch.cat([text_proj, torch.zeros_like(img_proj)], dim=-1)
img_token = torch.cat([torch.zeros_like(text_proj), img_proj], dim=-1)

tokens = torch.stack([text_token, img_token], dim=1)
tokens = tokens + self.pos_embedding

fused_tokens = self.fusion_transformer(tokens)
fused_rep = fused_tokens[:, 0, :]
logits = self.classifier(fused_rep)
```

Keempat blok kode tersebut sudah cukup untuk mewakili penjelasan mengenai alur utama model multimodal, tanpa perlu menampilkan keseluruhan file implementasi.
