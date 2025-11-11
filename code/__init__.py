"""
Multimodal Meme Classification Package
Klasifikasi Self-harm vs Non Self-harm menggunakan CLIP + Sentinet

Modules:
- data_loader: Dataset loader dan preprocessing
- model: Model multimodal dengan intermediate fusion
- train: Training loop dengan early stopping
- evaluate: Evaluation metrics dan visualisasi
- main_multimodal: Main script untuk menjalankan pipeline
"""

__version__ = "1.0.0"
__author__ = "Elsa Elisa Yohana Sianturi"
__description__ = "Multimodal Meme Classification: Self-harm Detection"

# Import fungsi utama untuk kemudahan akses
from .data_loader import load_dataset, get_processors, MultimodalMemeDataset
from .model import MultimodalMemeClassifier, count_parameters
from .train import Trainer, EarlyStopping
from .evaluate import (
    evaluate_model,
    print_evaluation_metrics,
    plot_confusion_matrix,
    plot_training_history,
    predict_samples,
    visualize_sample_predictions
)

__all__ = [
    # Data loading
    'load_dataset',
    'get_processors',
    'MultimodalMemeDataset',
    
    # Model
    'MultimodalMemeClassifier',
    'count_parameters',
    
    # Training
    'Trainer',
    'EarlyStopping',
    
    # Evaluation
    'evaluate_model',
    'print_evaluation_metrics',
    'plot_confusion_matrix',
    'plot_training_history',
    'predict_samples',
    'visualize_sample_predictions',
]
