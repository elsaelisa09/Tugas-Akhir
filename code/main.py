# Import libraries
import torch
import numpy as np
import random
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import modules
from data_loader import load_dataset, get_processors
from model import MultimodalMemeClassifier, count_parameters
from train import Trainer
from evaluate import (
    evaluate_model, 
    print_evaluation_metrics,
    plot_confusion_matrix,
    plot_training_history,
    predict_samples,
    visualize_sample_predictions
)


def set_seed(seed=42):
    """
    Set random seed untuk reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    # Set random seed
    SEED = 42
    set_seed(SEED)
    print(f"\n✓ Random seed set to {SEED}")
    
    # Paths
    CSV_PATH = "../Dataset/labels.csv"
    IMAGE_DIR = "../Dataset/images"
    SAVE_DIR = "./checkpoints"
    RESULTS_DIR = "./results"
    
    # Create directories
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 12
    PATIENCE = 5  # Early stopping 
    DROPOUT_RATE = 0.3
    VAL_SPLIT = 0.2
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
 
    print("HYPERPARAMETERS")
    print(f"Batch Size:      {BATCH_SIZE}")
    print(f"Learning Rate:   {LEARNING_RATE}")
    print(f"Num Epochs:      {NUM_EPOCHS}")
    print(f"Early Stopping:  {PATIENCE} epochs")
    print(f"Dropout Rate:    {DROPOUT_RATE}")
    print(f"Val Split:       {VAL_SPLIT * 100}%")

    
    
    # LOAD DATA 
 
    print(" LOAD DATA")

    
    # Get processors (CLIP dan sentinet tokenizer)
    clip_processor, text_tokenizer = get_processors()
    
    # Load dataset
    train_loader, val_loader, class_counts = load_dataset(
        csv_path=CSV_PATH,
        image_dir=IMAGE_DIR,
        clip_processor=clip_processor,
        text_tokenizer=text_tokenizer,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        random_state=SEED
    )
    
    print(f"\n Data loaded successfully!")
    
 
    print("BUILDING MODEL")
    
    
    model = MultimodalMemeClassifier(
        num_classes=2,
        dropout_rate=DROPOUT_RATE,
        freeze_encoders=False  # Fine-tune semua layer
    )
    
    # Count parameters
    trainable_params, total_params = count_parameters(model)
    print(f"\n{'='*70}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"{'='*70}\n")
    
    
    # TRAINING 
    print("TRAINING MODEL")

    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        save_dir=SAVE_DIR
    )
    
    # Start training
    history = trainer.train()
    
    # Load best model
    trainer.load_best_model()
    
    # Save training history plot
    plot_training_history(
        history, 
        save_path=os.path.join(RESULTS_DIR, 'training_history.png')
    )
    
    
    #  EVALUATION 
    print("EVALUATION ON VALIDATION SET")

    
    # Evaluate on validation set
    val_results = evaluate_model(
        model=model,
        data_loader=val_loader,
        device=device,
        criterion=torch.nn.CrossEntropyLoss()
    )
    
    # Print metrics
    print_evaluation_metrics(val_results)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        val_results['confusion_matrix'],
        class_names=['Non Self-harm', 'Self-harm'],
        save_path=os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    )
    
    # Predict samples
    samples_df = predict_samples(
        model=model,
        data_loader=val_loader,
        device=device,
        num_samples=20,
        class_names=['Non Self-harm', 'Self-harm']
    )
    
    # Save samples to CSV
    samples_df.to_csv(
        os.path.join(RESULTS_DIR, 'sample_predictions.csv'),
        index=False
    )
    print(f" Sample predictions saved to {RESULTS_DIR}/sample_predictions.csv")
    
    # Visualize sample predictions
    visualize_sample_predictions(
        model=model,
        dataset=val_loader.dataset,
        device=device,
        image_dir=IMAGE_DIR,
        num_samples=6,
        class_names=['Non Self-harm', 'Self-harm'],
        save_path=os.path.join(RESULTS_DIR, 'sample_predictions_visual.png')
    )
    
    
    # SUMMARY 
    print("TRAINING & EVALUATION COMPLETED")
    print(f"\n Best model saved at: {trainer.best_model_path}")
    print(f" Results saved in: {RESULTS_DIR}/")
    print(f"\nFinal Metrics:")
    print(f"  - Accuracy:  {val_results['accuracy']*100:.2f}%")
    print(f"  - Precision: {val_results['precision']*100:.2f}%")
    print(f"  - Recall:    {val_results['recall']*100:.2f}%")
    print(f"  - F1-Score:  {val_results['f1_score']*100:.2f}%")
    
    return model, history, val_results


if __name__ == "__main__":
    try:
        model, history, results = main()
        print("\n Pipeline completed successfully! \n")
    except Exception as e:
        print(f"\n Error occurred: {e}")
        import traceback
        traceback.print_exc()
