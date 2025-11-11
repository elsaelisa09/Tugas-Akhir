import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
from datetime import datetime


class EarlyStopping:
    """
    Early Stopping untuk menghentikan training jika validation loss tidak membaik
    """
    def __init__(self, patience=5, min_delta=0, verbose=True):
        """
        Args:
            patience: Berapa epoch untuk menunggu sebelum stop
            min_delta: Minimum perubahan untuk dianggap sebagai improvement
            verbose: Print message atau tidak
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss, epoch):
        """
        Cek apakah perlu early stopping
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0


class Trainer:
    """
    Trainer class untuk handle training loop
    """
    def __init__(self, model, train_loader, val_loader, device, 
                 learning_rate=2e-5, num_epochs=12, patience=5,
                 save_dir="./checkpoints"):
        """
        Args:
            model: Model PyTorch
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validation
            device: Device (cuda/cpu)
            learning_rate: Learning rate untuk optimizer
            num_epochs: Jumlah epoch maksimum
            patience: Patience untuk early stopping
            save_dir: Directory untuk menyimpan model checkpoint
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        
        # Buat direktori jika belum ada
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss function: CrossEntropyLoss
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer: AdamW
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01  # Weight decay untuk regularisasi
        )
        
        # Learning rate scheduler 
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience, verbose=True)
        
        # History untuk tracking
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        self.best_val_loss = float('inf')
        self.best_model_path = None
        
    def train_epoch(self):
        """
        Training untuk satu epoch
        Returns: average loss dan accuracy
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch in pbar:
            # Move data to device
            images = batch['image'].to(self.device)
            text_ids = batch['text_input_ids'].to(self.device)
            attention_mask = batch['text_attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, text_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping untuk stabilitas
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Hitung accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self):
        """
        Validation untuk satu epoch
        Returns: average loss dan accuracy
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for batch in pbar:
                # Move data to device
                images = batch['image'].to(self.device)
                text_ids = batch['text_input_ids'].to(self.device)
                attention_mask = batch['text_attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images, text_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                # Hitung accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """
        Simpan model checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }
        
        if is_best:
            # Simpan best model
            path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, path)
            self.best_model_path = path
            print(f"✓ Best model saved to {path}")
        
        # Simpan checkpoint terbaru
        latest_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
    
    def train(self):
        """
        Main training loop
        """
      
        print("TRAINING")
        print(f"Device: {self.device}")
        print(f"Number of epochs: {self.num_epochs}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        start_time = datetime.now()
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc = self.validate_epoch()
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Simpan ke history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Print summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.2e}")
            
            # Simpan best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self.save_checkpoint(epoch, val_loss, is_best=False)
            
            # Early stopping check
            self.early_stopping(val_loss, epoch)
            if self.early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"Best epoch was: {self.early_stopping.best_epoch}")
                print(f"Best val loss: {self.early_stopping.best_loss:.4f}")
                break
        
        # Training selesai
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("TRAINING COMPLETED")
        print(f"Total training time: {duration}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best model saved at: {self.best_model_path}")
        print(f"{'='*60}\n")
        
        return self.history
    
    def load_best_model(self):
        """
        Load best model dari checkpoint
        """
        if self.best_model_path and os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f" Loaded best model from {self.best_model_path}")
            return True
        else:
            print(" Best model not found!")
            return False


if __name__ == "__main__":
    """
    Testing training script dengan dummy data
    """
    print("Testing training script with dummy data...\n")
    
    from model import MultimodalMemeClassifier
    from torch.utils.data import TensorDataset, DataLoader
    
    # Create dummy data
    dummy_images = torch.randn(100, 3, 224, 224)
    dummy_text_ids = torch.randint(0, 1000, (100, 128))
    dummy_attention = torch.ones(100, 128)
    dummy_labels = torch.randint(0, 2, (100,))
    
    train_dataset = TensorDataset(dummy_images, dummy_text_ids, dummy_attention, dummy_labels)
    val_dataset = TensorDataset(dummy_images[:20], dummy_text_ids[:20], 
                                dummy_attention[:20], dummy_labels[:20])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalMemeClassifier(num_classes=2, dropout_rate=0.3)
    
    # Note: This will fail because dummy data format doesn't match
    # real data structure. Use main.py for real training.
    print(" Training script structure is ready!")
    print("Use main.py to run actual training with real data.")
