import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
from tqdm import tqdm
import pandas as pd
from PIL import Image
import os


def evaluate_model(model, data_loader, device, criterion=None):
    """
    Evaluasi model pada dataset
    
    Args:
        model: Model yang akan dievaluasi
        data_loader: DataLoader untuk dataset
        device: Device (cuda/cpu)
        criterion: Loss function (optional)
    
    Returns:
        Dictionary berisi metrik evaluasi
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    total_loss = 0
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating")
        
        for batch in pbar:
            # Move data to device
            images = batch['image'].to(device)
            text_ids = batch['text_input_ids'].to(device)
            attention_mask = batch['text_attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(images, text_ids, attention_mask)
            
            # Calculate loss if criterion provided
            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probs,
        'avg_loss': total_loss / len(data_loader) if criterion else None
    }
    
    return results


def print_evaluation_metrics(results, class_names=['Non Self-harm', 'Self-harm']):
    """
    Print metrik evaluasi dengan format yang rapi
    """
    print("EVALUATION METRICS")
    print(f"Accuracy:  {results['accuracy']*100:.2f}%")
    print(f"Precision: {results['precision']*100:.2f}%")
    print(f"Recall:    {results['recall']*100:.2f}%")
    print(f"F1-Score:  {results['f1_score']*100:.2f}%")
    
    if results['avg_loss'] is not None:
        print(f"Avg Loss:  {results['avg_loss']:.4f}")
    
    print("Classification Report:")
    report = classification_report(
        results['labels'], 
        results['predictions'],
        target_names=class_names,
        digits=4
    )
    print(report)
    
    print("Confusion Matrix:")
    print(f"{'':>15} {'Predicted':^30}")
    print(f"{'':>15} {class_names[0]:>15} {class_names[1]:>15}")
    print(f"{'Actual':>15} {'-'*30:>30}")
    for i, class_name in enumerate(class_names):
        print(f"{class_name:>15}", end="")
        for j in range(len(class_names)):
            print(f"{results['confusion_matrix'][i][j]:>15}", end="")
        print()


def plot_confusion_matrix(confusion_matrix, class_names=['Non Self-harm', 'Self-harm'], 
                         save_path='confusion_matrix.png'):
    """
    Visualisasi confusion matrix
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Confusion matrix saved to {save_path}")
    plt.close()


def plot_training_history(history, save_path='training_history.png'):
    """
    Visualisasi training history (loss dan accuracy)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot Loss
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=6)
    axes[0].plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=6)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy', linewidth=2, markersize=6)
    axes[1].plot(epochs, history['val_acc'], 'r-s', label='Val Accuracy', linewidth=2, markersize=6)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history plot saved to {save_path}")
    plt.close()


def predict_samples(model, data_loader, device, num_samples=10, 
                   class_names=['Non Self-harm', 'Self-harm']):
    """
    Prediksi beberapa sampel dan tampilkan hasilnya
    
    Args:
        model: Model yang sudah di-train
        data_loader: DataLoader
        device: Device (cuda/cpu)
        num_samples: Jumlah sampel yang akan ditampilkan
        class_names: Nama-nama kelas
    
    Returns:
        DataFrame dengan hasil prediksi
    """
    model.eval()
    
    samples_data = []
    collected = 0
    
    with torch.no_grad():
        for batch in data_loader:
            if collected >= num_samples:
                break
            
            # Move data to device
            images = batch['image'].to(device)
            text_ids = batch['text_input_ids'].to(device)
            attention_mask = batch['text_attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(images, text_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Collect samples
            batch_size = images.size(0)
            for i in range(min(batch_size, num_samples - collected)):
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                confidence = probs[i][pred_label].item()
                
                samples_data.append({
                    'True Label': class_names[true_label],
                    'Predicted Label': class_names[pred_label],
                    'Confidence': f"{confidence*100:.2f}%",
                    'Correct': '✓' if true_label == pred_label else '✗'
                })
                
                collected += 1
                if collected >= num_samples:
                    break
    
    # Create DataFrame
    df = pd.DataFrame(samples_data)
    
    print("\n" + "="*60)
    print(f"SAMPLE PREDICTIONS (First {num_samples} samples)")
    print("="*60)
    print(df.to_string(index=True))
    print("="*60 + "\n")
    
    return df


def visualize_sample_predictions(model, dataset, device, image_dir, num_samples=6,
                                 class_names=['Non Self-harm', 'Self-harm'],
                                 save_path='sample_predictions.png'):
    """
    Visualisasi prediksi dengan gambar asli
    
    Args:
        model: Model yang sudah di-train
        dataset: Dataset object (bukan DataLoader)
        device: Device
        image_dir: Path ke folder gambar
        num_samples: Jumlah sampel
        class_names: Nama kelas
        save_path: Path untuk save gambar
    """
    model.eval()
    
    # Ambil random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            if idx >= num_samples:
                break
            
            # Get sample
            sample = dataset[sample_idx]
            
            # Prepare input
            image = sample['image'].unsqueeze(0).to(device)
            text_ids = sample['text_input_ids'].unsqueeze(0).to(device)
            attention_mask = sample['text_attention_mask'].unsqueeze(0).to(device)
            true_label = sample['label'].item()
            
            # Predict
            output = model(image, text_ids, attention_mask)
            probs = torch.softmax(output, dim=1)
            pred_label = torch.argmax(output, dim=1).item()
            confidence = probs[0][pred_label].item()
            
            # Load original image for visualization
            image_filename = dataset.dataframe.iloc[sample_idx]['File_Name']
            image_path = os.path.join(image_dir, image_filename)
            
            try:
                original_image = Image.open(image_path).convert('RGB')
            except:
                # Create dummy image if loading fails
                original_image = Image.new('RGB', (224, 224), color='gray')
            
            # Plot
            axes[idx].imshow(original_image)
            axes[idx].axis('off')
            
            # Title dengan warna berbeda untuk correct/incorrect
            color = 'green' if true_label == pred_label else 'red'
            title = f"True: {class_names[true_label]}\n"
            title += f"Pred: {class_names[pred_label]}\n"
            title += f"Conf: {confidence*100:.1f}%"
            axes[idx].set_title(title, fontsize=10, color=color, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(len(indices), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Sample predictions visualization saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    """
    Testing evaluation functions
    """
    print("Testing evaluation script...")
    
    # Dummy results for testing
    dummy_results = {
        'accuracy': 0.85,
        'precision': 0.83,
        'recall': 0.87,
        'f1_score': 0.85,
        'confusion_matrix': np.array([[45, 5], [10, 40]]),
        'predictions': np.array([0, 1, 1, 0, 1]),
        'labels': np.array([0, 1, 0, 0, 1]),
        'probabilities': np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.8, 0.2], [0.1, 0.9]]),
        'avg_loss': 0.35
    }
    
    print_evaluation_metrics(dummy_results)
    
    # Test plotting
    dummy_history = {
        'train_loss': [0.6, 0.5, 0.4, 0.35, 0.3],
        'val_loss': [0.65, 0.55, 0.48, 0.42, 0.4],
        'train_acc': [70, 75, 80, 82, 85],
        'val_acc': [68, 73, 78, 80, 82]
    }
    
    print("\n Evaluation functions are ready!")
    print("Use main.py to run full evaluation with trained model.")
