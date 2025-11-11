"""
Data Loader untuk Dataset Multimodal (Gambar + Teks)
Membaca CSV, gambar, dan teks untuk klasifikasi Self-harm vs Non Self-harm
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from transformers import CLIPProcessor, AutoTokenizer
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import random


class MultimodalMemeDataset(Dataset):
    """
    Dataset custom untuk meme multimodal (gambar + teks)
    """
    def __init__(self, dataframe, image_dir, clip_processor, text_tokenizer, max_length=128, augment=False):
        """
        Args:
            dataframe: DataFrame dengan kolom 'File_Name', 'Teks_Terlihat', 'Label'
            image_dir: Path ke folder gambar
            clip_processor: Processor dari CLIP untuk preprocessing gambar
            text_tokenizer: Tokenizer dari model text (sentinet/suicidality)
            max_length: Panjang maksimum sequence untuk teks
            augment: Aktifkan augmentasi data (untuk training set)
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.clip_processor = clip_processor
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length
        self.augment = augment
        
        # Encode labels: "Self-harm" -> 1, "Non Self-harm" -> 0
        self.label_map = {"Self-harm": 1, "Non Self-harm": 0}
        
        # Augmentasi gambar untuk training (sebelum CLIP processor)
        if self.augment:
            self.image_augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            ])
        else:
            self.image_augmentation = transforms.Compose([
                transforms.Resize((224, 224)),
            ])
    
    def __len__(self):
        return len(self.dataframe)
    
    def augment_text(self, text):
        """
        Augmentasi teks sederhana (random synonym replacement atau random deletion)
        """
        if not self.augment or not text or len(text) < 5:
            return text
        
        # Random choice: 70% tidak diubah, 30% diubah
        if random.random() > 0.3:
            return text
        
        words = text.split()
        if len(words) < 2:
            return text
        
        # Teknik 1: Random Deletion (hapus kata secara random)
        if random.random() < 0.5 and len(words) > 3:
            num_to_delete = max(1, len(words) // 10)  # Hapus 10% kata
            indices_to_delete = random.sample(range(len(words)), num_to_delete)
            words = [w for i, w in enumerate(words) if i not in indices_to_delete]
        
        # Teknik 2: Random Swap (tukar posisi 2 kata)
        elif len(words) > 2:
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def __getitem__(self, idx):
        """
        Mengambil satu sampel data multimodal
        Returns: dict dengan keys 'image', 'text', 'label'
        """
        row = self.dataframe.iloc[idx]
        
        # 1. Load dan preprocess gambar dengan CLIP + Augmentasi
        image_path = os.path.join(self.image_dir, row['File_Name'])
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Apply augmentasi gambar
            image = self.image_augmentation(image)
            
            # CLIP processor akan mengubah gambar ke tensor dan normalisasi
            image_inputs = self.clip_processor(images=image, return_tensors="pt")
            # Ambil pixel_values dan squeeze dimension pertama (batch)
            image_tensor = image_inputs['pixel_values'].squeeze(0)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Jika error, buat dummy image (tensor kosong)
            image_tensor = torch.zeros(3, 224, 224)
        
        # 2. Tokenisasi teks dengan model sentinet + Augmentasi
        text = str(row['Teks_Terlihat']) if pd.notna(row['Teks_Terlihat']) else ""
        
        # Apply augmentasi teks (hanya untuk training)
        text = self.augment_text(text)
        
        text_inputs = self.text_tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Squeeze dimension batch (dari [1, seq_len] jadi [seq_len])
        text_input_ids = text_inputs['input_ids'].squeeze(0)
        text_attention_mask = text_inputs['attention_mask'].squeeze(0)
        
        # 3. Encode label
        label = self.label_map.get(row['Label'], 0)
        
        return {
            'image': image_tensor,
            'text_input_ids': text_input_ids,
            'text_attention_mask': text_attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_dataset(csv_path, image_dir, clip_processor, text_tokenizer, 
                 batch_size=16, val_split=0.2, random_state=42):
    """
    Load dataset dari CSV dan split menjadi train/val
    
    Args:
        csv_path: Path ke file labels.csv
        image_dir: Path ke folder images
        clip_processor: CLIP processor untuk gambar
        text_tokenizer: Tokenizer untuk teks
        batch_size: Batch size untuk DataLoader
        val_split: Proporsi validation set (default 0.2 = 20%)
        random_state: Seed untuk reproducibility
    
    Returns:
        train_loader, val_loader, class_counts
    """
    
    # Load CSV
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter baris yang memiliki label valid
    df = df[df['Label'].isin(['Self-harm', 'Non Self-harm'])].copy()
    
    # Cek distribusi kelas
    class_counts = df['Label'].value_counts()
    print(f"\nDistribusi kelas:")
    print(class_counts)
    print(f"\nTotal sampel: {len(df)}")
    
    # Split train/validation dengan stratifikasi
    train_df, val_df = train_test_split(
        df, 
        test_size=val_split, 
        stratify=df['Label'], 
        random_state=random_state
    )
    
    print(f"\nTrain set: {len(train_df)} sampel")
    print(f"Validation set: {len(val_df)} sampel")
    
    # Buat dataset objects
    train_dataset = MultimodalMemeDataset(
        train_df, image_dir, clip_processor, text_tokenizer, augment=True
    )
    
    val_dataset = MultimodalMemeDataset(
        val_df, image_dir, clip_processor, text_tokenizer, augment=False
    )
    
    # Buat DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set 0 untuk Windows, bisa dinaikkan di Linux/Mac
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, class_counts


def get_processors():
    """
    Inisialisasi CLIP processor dan text tokenizer
    
    Returns:
        clip_processor, text_tokenizer
    """
    print("Loading CLIP processor...")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    print("Loading sentinet/suicidality tokenizer...")
    text_tokenizer = AutoTokenizer.from_pretrained("sentinet/suicidality")
    
    return clip_processor, text_tokenizer


if __name__ == "__main__":
    """
    Testing data loader
    """
    # Path ke data
    csv_path = "../Dataset/labels.csv"
    image_dir = "../Dataset/images"
    
    # Load processors
    clip_processor, text_tokenizer = get_processors()
    
    # Load dataset
    train_loader, val_loader, class_counts = load_dataset(
        csv_path, 
        image_dir, 
        clip_processor, 
        text_tokenizer,
        batch_size=8
    )
    
    # Test satu batch
    print("\n=== Testing Data Loader ===")
    for batch in train_loader:
        print(f"Image shape: {batch['image'].shape}")
        print(f"Text input_ids shape: {batch['text_input_ids'].shape}")
        print(f"Text attention_mask shape: {batch['text_attention_mask'].shape}")
        print(f"Labels shape: {batch['label'].shape}")
        print(f"Labels: {batch['label']}")
        break
