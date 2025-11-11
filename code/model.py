import torch
import torch.nn as nn
from transformers import CLIPModel, AutoModel


class MultimodalMemeClassifier(nn.Module):
    """
    Model multimodal dengan intermediate fusion
    - Image encoder: CLIP ViT-B/32
    - Text encoder: sentinet/suicidality
    - Fusion: Concatenate + MLP
    - Classifier: Fully Connected Layer dengan Dropout
    """
    
    def __init__(self, num_classes=2, dropout_rate=0.3, freeze_encoders=False):
        """
        Args:
            num_classes: Jumlah kelas output (2: Self-harm, Non Self-harm)
            dropout_rate: Dropout rate untuk regularisasi (default 0.3)
            freeze_encoders: Jika True, freeze pretrained encoders (hanya train fusion layer)
        """
        super(MultimodalMemeClassifier, self).__init__()
        
        #  IMAGE ENCODER: CLIP 
        print("Loading CLIP model for image encoding...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # Dimensi output CLIP vision encoder
        self.image_embed_dim = self.clip_model.config.vision_config.hidden_size  # 768
        
        # Freeze CLIP parameters jika diperlukan
        if freeze_encoders:
            for param in self.clip_model.vision_model.parameters():
                param.requires_grad = False
        
        # TEXT ENCODER: sentinet/suicidality 
        print("Loading sentinet/suicidality model for text encoding...")
        self.text_model = AutoModel.from_pretrained("sentinet/suicidality")
        
        # Dimensi output text encoder ( 768 untuk BERT-based)
        self.text_embed_dim = self.text_model.config.hidden_size  # 768
        
        # Freeze text encoder parameters jika diperlukan
        if freeze_encoders:
            for param in self.text_model.parameters():
                param.requires_grad = False
        
        #  FUSION LAYER: Concatenate + MLP 
        # Total dimensi setelah concatenate
        self.fused_dim = self.image_embed_dim + self.text_embed_dim  # 768 + 768 = 1536
        
        # Multi-Layer Perceptron untuk fusion
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        #  CLASSIFIER HEAD 
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        print(f"\nModel Architecture:")
        print(f"  - Image embedding dim: {self.image_embed_dim}")
        print(f"  - Text embedding dim: {self.text_embed_dim}")
        print(f"  - Fused dim: {self.fused_dim}")
        print(f"  - Dropout rate: {dropout_rate}")
        print(f"  - Freeze encoders: {freeze_encoders}")
        print(f"  - Output classes: {num_classes}")
    
    def forward(self, image, text_input_ids, text_attention_mask):
        """
        Forward pass untuk model multimodal
        
        Args:
            image: Tensor gambar [batch_size, 3, 224, 224]
            text_input_ids: Input IDs untuk text [batch_size, seq_len]
            text_attention_mask: Attention mask untuk text [batch_size, seq_len]
        
        Returns:
            logits: Output klasifikasi [batch_size, num_classes]
        """
        
        #  IMAGE ENCODING 
        # Extract image features menggunakan CLIP vision encoder
        vision_outputs = self.clip_model.vision_model(pixel_values=image)
        
        # Ambil pooled output (CLS token equivalent untuk vision)
        # Shape: [batch_size, 768]
        image_embeds = vision_outputs.pooler_output
        
        # TEXT ENCODING 
        # Extract text features menggunakan sentinet model
        text_outputs = self.text_model(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )
        
        # Sentinet tidak punya pooler_output, gunakan mean pooling
        # Ambil last_hidden_state: [batch_size, seq_len, hidden_size]
        text_embeds = text_outputs.last_hidden_state
        
        # Mean pooling dengan attention mask
        attention_mask_expanded = text_attention_mask.unsqueeze(-1).expand(text_embeds.size()).float()
        sum_embeddings = torch.sum(text_embeds * attention_mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        text_embeds = sum_embeddings / sum_mask  # Shape: [batch_size, 768]
        
        # INTERMEDIATE FUSION 
        # Concatenate image dan text embeddings
        # Shape: [batch_size, 1536]
        fused_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        
        # Pass melalui MLP fusion layer
        # Shape: [batch_size, 256]
        fused_features = self.fusion_mlp(fused_embeds)
        
        #  CLASSIFICATION 
        # Pass melalui classifier head
        # Shape: [batch_size, num_classes]
        logits = self.classifier(fused_features)
        
        return logits
    
    def get_embedding(self, image, text_input_ids, text_attention_mask):
        """
        Mendapatkan fused embedding sebelum classification layer
        Berguna untuk visualisasi atau analisis
        
        Returns:
            fused_features: [batch_size, 256]
        """
        with torch.no_grad():
            # Image encoding
            vision_outputs = self.clip_model.vision_model(pixel_values=image)
            image_embeds = vision_outputs.pooler_output
            
            # Text encoding
            text_outputs = self.text_model(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask
            )
            # Mean pooling untuk text embeddings
            text_embeds = text_outputs.last_hidden_state
            attention_mask_expanded = text_attention_mask.unsqueeze(-1).expand(text_embeds.size()).float()
            sum_embeddings = torch.sum(text_embeds * attention_mask_expanded, dim=1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
            text_embeds = sum_embeddings / sum_mask
            
            # Fusion
            fused_embeds = torch.cat([image_embeds, text_embeds], dim=1)
            fused_features = self.fusion_mlp(fused_embeds)
            
        return fused_features


def count_parameters(model):
    """
    Hitung jumlah parameter yang trainable
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    """
    Testing model architecture
    """
    print("=== Testing Multimodal Model ===\n")
    
    # Inisialisasi model
    model = MultimodalMemeClassifier(
        num_classes=2,
        dropout_rate=0.3,
        freeze_encoders=False
    )
    
    # Hitung parameter
    trainable_params, total_params = count_parameters(model)
    print(f"\n=== Model Parameters ===")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    # Test forward pass dengan dummy data
    print("\n=== Testing Forward Pass ===")
    batch_size = 4
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    dummy_text_ids = torch.randint(0, 1000, (batch_size, 128))
    dummy_attention_mask = torch.ones(batch_size, 128)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_image, dummy_text_ids, dummy_attention_mask)
    
    print(f"Input image shape: {dummy_image.shape}")
    print(f"Input text_ids shape: {dummy_text_ids.shape}")
    print(f"Output logits shape: {outputs.shape}")
    print(f"Output logits example:\n{outputs[:2]}")
    
    # Test embedding extraction
    print("\n=== Testing Embedding Extraction ===")
    embeddings = model.get_embedding(dummy_image, dummy_text_ids, dummy_attention_mask)
    print(f"Fused embeddings shape: {embeddings.shape}")
    
    print("\n Model test completed successfully!")
