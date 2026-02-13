"""
Main training script for the Intermediate Fusion model.
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel

from data_loader import MultiModalDataset, collate_batch
from model2 import CLIPElectraFusionNoTransformer, EarlyStopping
from train import train_one_epoch, setup_optimizer, setup_scheduler
from evaluation import (
    evaluate, 
    plot_confusion_matrix, 
    plot_training_history,
    print_classification_report,
    analyze_model_parameters
)


class Config:
    """Configuration class for training parameters."""
    
    CURRENT_DIR = os.getcwd()
    RESULTS_DIR = os.path.join(CURRENT_DIR, 'results')
    IMAGES_DIR = os.path.join(CURRENT_DIR, 'balanced-dataset-final')
    LABELS_CSV = os.path.join(CURRENT_DIR, 'CSV-balanced-data-pseudo-labeling.csv')
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 16
    EPOCHS = 12
    LR_PRETRAIN = 1e-5
    LR_HEAD = 5e-4
    MAX_LEN = 128
    NUM_CLASSES = 2
    IMAGE_SIZE = 224
    SEED = 42
    NUM_WORKERS = 2
    PATIENCE = 3
    
    FUSION_IMG_DIM = 512
    FUSION_TEXT_DIM = 256


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(config):

    assert os.path.exists(config.LABELS_CSV), f'CSV not found: {config.LABELS_CSV}'
    assert os.path.isdir(config.IMAGES_DIR), f'Images folder not found: {config.IMAGES_DIR}'
    
    labels_df = pd.read_csv(config.LABELS_CSV)
    label_mapping = {'NON-SELF-HARM': 0, 'SELF-HARM': 1}
    
    if 'Label Akhir' not in labels_df.columns or 'filename' not in labels_df.columns:
        raise ValueError('CSV must have columns: Label Akhir, filename, and Teks Terlihat (optional)')
    
    labels_df = labels_df[labels_df['Label Akhir'].isin(label_mapping.keys())].copy()
    labels_df['Label'] = labels_df['Label Akhir'].map(label_mapping)
    
    print(f'Total samples after filtering: {len(labels_df)}')
    print(f'Label distribution:\n{labels_df["Label"].value_counts().sort_index()}')
    
    train_df, val_df = train_test_split(
        labels_df, 
        test_size=0.2, 
        random_state=config.SEED, 
        stratify=labels_df['Label']
    )
    print(f'Train: {len(train_df)} | Val: {len(val_df)}')
    
    clip_model_name = 'openai/clip-vit-base-patch32'
    electra_model_name = 'sentinet/suicidality'
    
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    electra_tokenizer = AutoTokenizer.from_pretrained(electra_model_name)
    
    train_ds = MultiModalDataset(
        train_df, config.IMAGES_DIR, electra_tokenizer, 
        clip_processor, max_len=config.MAX_LEN, is_train=True
    )
    val_ds = MultiModalDataset(
        val_df, config.IMAGES_DIR, electra_tokenizer, 
        clip_processor, max_len=config.MAX_LEN, is_train=False
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True, 
        num_workers=config.NUM_WORKERS, collate_fn=collate_batch, 
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False, 
        num_workers=config.NUM_WORKERS, collate_fn=collate_batch, 
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, clip_processor, electra_tokenizer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Intermediate Fusion Model')
    parser.add_argument('--images_dir', type=str, default=None, 
                        help='Path to images directory')
    parser.add_argument('--labels_csv', type=str, default=None, 
                        help='Path to labels CSV file')
    parser.add_argument('--batch_size', type=int, default=None, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=None, 
                        help='Number of training epochs')
    parser.add_argument('--lr_pretrain', type=float, default=None, 
                        help='Learning rate for pretrained layers')
    parser.add_argument('--lr_head', type=float, default=None, 
                        help='Learning rate for head layers')
    args = parser.parse_args()
    
    config = Config()
    if args.images_dir:
        config.IMAGES_DIR = args.images_dir
    if args.labels_csv:
        config.LABELS_CSV = args.labels_csv
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.lr_pretrain:
        config.LR_PRETRAIN = args.lr_pretrain
    if args.lr_head:
        config.LR_HEAD = args.lr_head
    
    set_seed(config.SEED)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    print(f'Device: {config.DEVICE}')
    print(f'Results directory: {config.RESULTS_DIR}')
    print(f'CSV path: {config.LABELS_CSV}')
    print(f'Images directory: {config.IMAGES_DIR}\n')
    
    train_loader, val_loader, clip_processor, electra_tokenizer = load_data(config)
    
    print('\nLoading pretrained models...')
    clip_model_name = 'openai/clip-vit-base-patch32'
    electra_model_name = 'sentinet/suicidality'
    
    clip_model = CLIPModel.from_pretrained(clip_model_name)
    electra_model = AutoModel.from_pretrained(electra_model_name)
    
    print('Initializing fusion model...')
    model = CLIPElectraFusionNoTransformer(
        clip_model=clip_model,
        electra_model=electra_model,
        fusion_img_dim=config.FUSION_IMG_DIM,
        fusion_text_dim=config.FUSION_TEXT_DIM,
        num_classes=config.NUM_CLASSES,
        freeze_encoders=True
    )
    model = model.to(config.DEVICE)
    
    analyze_model_parameters(model)
    
    optimizer = setup_optimizer(model, config.LR_PRETRAIN, config.LR_HEAD)
    scheduler = setup_scheduler(optimizer)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(patience=config.PATIENCE, mode='max')
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lrs': []
    }
    
    best_model_path = os.path.join(config.RESULTS_DIR, 'best_multimodal2.pth')
    best_f1 = -1
    
    print('\nStarting training...\n')
    
    for epoch in range(1, config.EPOCHS + 1):
        print(f'Epoch {epoch}/{config.EPOCHS}')
        print('-' * 60)
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, config.DEVICE
        )
        
        val_loss, val_acc, p, r, f1, cm, val_preds, val_trues = evaluate(
            model, val_loader, criterion, config.DEVICE
        )
        
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        history['lrs'].append(current_lr)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}')
        print(f'Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}')
        print(f'Learning Rate: {current_lr:.2e}')
        print(f'Confusion Matrix:\n{cm}\n')
        
        improved = early_stopper.step(f1)
        
        if improved:
            print(f'Best model saved (F1 improved to {f1:.4f})')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': f1,
                'history': history
            }, best_model_path)
            best_f1 = f1
        else:
            print(f'No improvement ({early_stopper.num_bad}/{config.PATIENCE})')
        
        if early_stopper.should_stop:
            print('\nEarly stopping triggered!')
            break
        
        print()
    
    print(f'\nTraining completed. Best F1: {best_f1:.4f}')
    print(f'Best model saved to: {best_model_path}')
    
    print('\nLoading best model for final evaluation...')
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, val_acc, p, r, f1, cm, val_preds, val_trues = evaluate(
        model, val_loader, criterion, config.DEVICE
    )
    
    print('\nFinal Validation Results:')
    print(f'Accuracy:  {val_acc:.4f}')
    print(f'Precision: {p:.4f}')
    print(f'Recall:    {r:.4f}')
    print(f'F1 Score:  {f1:.4f}')
    
    print_classification_report(val_trues, val_preds)
    
    cm_path = os.path.join(config.RESULTS_DIR, 'confusion_matrix2.png')
    plot_confusion_matrix(cm, save_path=cm_path)
    
    history_path = os.path.join(config.RESULTS_DIR, 'training_history2.png')
    plot_training_history(history, save_path=history_path)
    
    print(f'\nPlots saved to {config.RESULTS_DIR}')


if __name__ == '__main__':
    main()
