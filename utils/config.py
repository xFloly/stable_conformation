from pathlib import Path
import torch

def get_config():
    """
        Returns a dictionary containing the configuration for the experiment.

        Returns:
            dict: Configuration parameters including model settings, dataset details,
                optimizer, scheduler, and other hyperparameters.
    """
    return {
        "task": "classification",  # (regression, classification)
        "batch_size": 64,  # Number of samples per batch
        "num_epochs": 100,  # Total number of epochs for training
        "optimizer": "AdamW",  # Type of optimizer to use (e.g., Adam, SGD, AdamW)
        "lr": 0.00001,  # Learning rate for the optimizer
        "weight_decay": 0.00001,  # Regularization term to prevent overfitting
        "scheduler": "CosineAnnealingLR",  # Type of learning rate scheduler
        "scheduler_t_max": 10,  # Num of epochs over which to decay the learning rate for scheduler
        "scheduler_eta_min": 0.0001,  # Minimum learning rate value for the scheduler
        "loss_fn": "weighted_mse",  # Loss function to use (e.g., mse, weighted_mse, kl_loss)
        "seed": 42,  # Random seed for reproducibility of experiments
        "num_workers": -1,  # Number of workers for DataLoader (-1 = use all available CPU cores)
        "train_data": "data/downloads_train",  # Path to the train dataset
        "test_data": "data/downloads_test",  # Path to the test dataset
        "model_name": "resnet_optical_flow",  # Name of the model architecture to use
        "model_name_log": "resnet_optical_flow_weighted_mse",  # Name of the model log file
        "model_basename": "model_",  # Base name for saving and loading model weight files
        "preload": "latest",  # Preload setting to load weights: "latest", "none", or specific point
        "dataset_path": "data/test_data",  # Path to the dataset directory
        "device": "cuda:0",  # Device to use for training and evaluation (cuda:0 or cpu)
        "preprocess_data_path": "data/full_preprocess_data",  # Path to data ready to use in training
        # RESNET PARAMS
    }


def get_config_eval():
    """
    Returns a dictionary containing the configuration for the validation process.

    Returns:
        dict: Configuration parameters including model settings and dataset details.
    """
    return {
        # Task and data handling
        "task": "regression",  # Task type (classification or regression)
        "batch_size": 64,  # Number of samples per batch
        "seed": 42,  # Random seed for reproducibility
        "num_workers": -1,  # Number of workers for DataLoader (-1 = use all available CPU cores)
        "test_data": "data/visualize_video_train",  # Path to the validation dataset
        "dataset_path": "data/visualize_video_train",  # Path to the dataset directory
        "preprocess_data_path": "data/preprocess_video_train",  # Path to preprocessed data

        # Model parameters
        "model_name": "resnet_optical_flow",  # Name of the model architecture
        "model_name_log": "resnet_optical_flow_mse",  # Name for log files
        "model_basename": "model_",  # Base name for saving/loading model weights
        "preload": "latest",  # Preload option for loading weights
        "device": "cuda:0",  # Device to use for validation (cuda or cpu)
        "model_depth": 18,  # Depth of ResNet model
        "num_of_classes": 2,  # Number of output classes
        "loss_fn": "mse",  # Loss function to use (e.g., mse, weighted_mse, kl_loss)

        # Classification-specific parameters
        "shape": 100,  # Grid size for classification task

        # Metrics
        "grid_size": 3,  # grid size for f1,recal,prec

        # logging
        "epochs_per_log": 100,  # Number of epochs between logging
    }