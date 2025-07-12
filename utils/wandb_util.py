import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import json
from typing import Dict, List, Optional, Any
import pandas as pd


class WandbLogger:
    """Weights & Biases logger for text classification experiments"""
    
    def __init__(self, 
                 project_name: str = "text-classification-imdb",
                 experiment_name: Optional[str] = None,
                 config: Optional[Dict] = None,
                 tags: Optional[List[str]] = None,
                 notes: Optional[str] = None):
        """
        Initialize wandb logger
        
        Args:
            project_name: Name of the wandb project
            experiment_name: Name of the experiment run
            config: Configuration dictionary to log
            tags: List of tags for the experiment
            notes: Notes about the experiment
        """
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config or {}
        self.tags = tags or []
        self.notes = notes
        self.run = None
        
    def init_wandb(self, model_type: str, resume: bool = False):
        """
        Initialize wandb run
        
        Args:
            model_type: Type of model being trained
            resume: Whether to resume from existing run
        """
        run_name = f"{model_type}_{self.experiment_name}" if self.experiment_name else model_type
        
        self.run = wandb.init(
            project=self.project_name,
            name=run_name,
            config=self.config,
            tags=self.tags + [model_type],
            notes=self.notes,
            resume=resume
        )
        
        print(f"🚀 Started wandb run: {self.run.name}")
        print(f"📊 View at: {self.run.url}")
        
    def log_config(self, config: Dict):
        """Log configuration parameters"""
        if self.run:
            wandb.config.update(config)
        
    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        """
        Log metrics to wandb
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Training step (optional)
        """
        if self.run:
            wandb.log(metrics, step=step)
    
    def log_model_architecture(self, model: torch.nn.Module, input_shape: tuple):
        """
        Log model architecture and parameters
        
        Args:
            model: PyTorch model
            input_shape: Shape of input tensor (batch_size, seq_len)
        """
        if not self.run:
            return
            
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Log model info
        self.log_metrics({
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/non_trainable_parameters": total_params - trainable_params
        })
        
        # Create a dummy input and log model graph (optional)
        try:
            dummy_input = torch.randint(0, 1000, input_shape)
            # Note: wandb.watch can be used here for gradient tracking
            # wandb.watch(model, log="all", log_freq=100)
        except Exception as e:
            print(f"Could not log model graph: {e}")
    
    def log_confusion_matrix(self, 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray, 
                           class_names: List[str] = None,
                           title: str = "Confusion Matrix"):
        """
        Log confusion matrix to wandb
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            title: Title for the plot
        """
        if not self.run:
            return
            
        if class_names is None:
            class_names = ["Negative", "Positive"]
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Log to wandb
        wandb.log({f"confusion_matrix/{title.lower().replace(' ', '_')}": wandb.Image(plt)})
        plt.close()
        
        # Also log as wandb confusion matrix
        wandb.log({f"confusion_matrix_wandb/{title.lower().replace(' ', '_')}": 
                  wandb.plot.confusion_matrix(
                      probs=None,
                      y_true=y_true,
                      preds=y_pred,
                      class_names=class_names)})
    
    def log_classification_report(self, 
                                y_true: np.ndarray, 
                                y_pred: np.ndarray,
                                class_names: List[str] = None,
                                prefix: str = ""):
        """
        Log detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            prefix: Prefix for metric names
        """
        if not self.run:
            return
            
        if class_names is None:
            class_names = ["Negative", "Positive"]
        
        # Generate classification report
        report = classification_report(y_true, y_pred, 
                                     target_names=class_names,
                                     output_dict=True)
        
        # Log individual class metrics
        for class_name in class_names:
            if class_name in report:
                metrics = report[class_name]
                for metric_name, value in metrics.items():
                    metric_key = f"{prefix}class_{class_name.lower()}_{metric_name}"
                    self.log_metrics({metric_key: value})
        
        # Log overall metrics
        for metric_type in ['macro avg', 'weighted avg']:
            if metric_type in report:
                metrics = report[metric_type]
                for metric_name, value in metrics.items():
                    metric_key = f"{prefix}{metric_type.replace(' ', '_')}_{metric_name}"
                    self.log_metrics({metric_key: value})
    
    def log_learning_curves(self, 
                          train_losses: List[float],
                          val_accuracies: List[float],
                          val_losses: List[float] = None):
        """
        Log learning curves
        
        Args:
            train_losses: List of training losses per epoch
            val_accuracies: List of validation accuracies per epoch
            val_losses: List of validation losses per epoch (optional)
        """
        if not self.run:
            return
        
        epochs = range(1, len(train_losses) + 1)
        
        # Create learning curves plot
        fig, axes = plt.subplots(1, 2 if val_losses else 1, figsize=(15, 5))
        
        if val_losses:
            # Loss plot
            axes[0].plot(epochs, train_losses, label='Training Loss', color='blue')
            axes[0].plot(epochs, val_losses, label='Validation Loss', color='red')
            axes[0].set_title('Training and Validation Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True)
            
            # Accuracy plot
            axes[1].plot(epochs, val_accuracies, label='Validation Accuracy', color='green')
            axes[1].set_title('Validation Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True)
        else:
            # Single plot
            ax = axes if isinstance(axes, plt.Axes) else axes[0]
            ax.plot(epochs, train_losses, label='Training Loss', color='blue')
            ax.plot(epochs, val_accuracies, label='Validation Accuracy', color='green')
            ax.set_title('Training Loss and Validation Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        wandb.log({"learning_curves": wandb.Image(fig)})
        plt.close()
    
    def log_sample_predictions(self, 
                             texts: List[str],
                             true_labels: List[int],
                             pred_labels: List[int],
                             pred_probs: List[float],
                             class_names: List[str] = None,
                             num_samples: int = 10):
        """
        Log sample predictions as a table
        
        Args:
            texts: List of text samples
            true_labels: List of true labels
            pred_labels: List of predicted labels
            pred_probs: List of prediction probabilities
            class_names: Names of classes
            num_samples: Number of samples to log
        """
        if not self.run:
            return
            
        if class_names is None:
            class_names = ["Negative", "Positive"]
        
        # Select random samples
        indices = np.random.choice(len(texts), 
                                 size=min(num_samples, len(texts)), 
                                 replace=False)
        
        # Create table data
        table_data = []
        for idx in indices:
            true_class = class_names[true_labels[idx]]
            pred_class = class_names[pred_labels[idx]]
            confidence = pred_probs[idx]
            correct = "✓" if true_labels[idx] == pred_labels[idx] else "✗"
            
            table_data.append([
                texts[idx][:100] + "..." if len(texts[idx]) > 100 else texts[idx],
                true_class,
                pred_class,
                f"{confidence:.3f}",
                correct
            ])
        
        # Create wandb table
        table = wandb.Table(
            columns=["Text", "True Label", "Predicted Label", "Confidence", "Correct"],
            data=table_data
        )
        
        wandb.log({"sample_predictions": table})
    
    def log_hyperparameters(self, hyperparams: Dict):
        """Log hyperparameters"""
        if self.run:
            wandb.config.update(hyperparams)
    
    def log_dataset_info(self, train_size: int, val_size: int, test_size: int):
        """Log dataset information"""
        if self.run:
            self.log_metrics({
                "dataset/train_size": train_size,
                "dataset/val_size": val_size,
                "dataset/test_size": test_size,
                "dataset/total_size": train_size + val_size + test_size
            })
    
    def save_model_artifact(self, 
                          model_path: str, 
                          model_name: str,
                          metadata: Dict = None):
        """
        Save model as wandb artifact
        
        Args:
            model_path: Path to the saved model
            model_name: Name for the artifact
            metadata: Additional metadata
        """
        if not self.run:
            return
            
        artifact = wandb.Artifact(
            name=model_name,
            type="model",
            metadata=metadata or {}
        )
        
        artifact.add_file(model_path)
        self.run.log_artifact(artifact)
        
        print(f"💾 Saved model artifact: {model_name}")
    
    def finish(self):
        """Finish wandb run"""
        if self.run:
            wandb.finish()
            print("🏁 Finished wandb run")


def compare_models_wandb(results: Dict[str, Dict], 
                        project_name: str = "text-classification-comparison"):
    """
    Create a comparison visualization of multiple models
    
    Args:
        results: Dictionary of model_name -> metrics
        project_name: Name of the wandb project
    """
    # Initialize wandb for comparison
    wandb.init(project=project_name, name="model_comparison")
    
    # Prepare data for comparison
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Create comparison table
    table_data = []
    for model in models:
        row = [model]
        for metric in metrics:
            row.append(results[model].get(metric, 0.0))
        table_data.append(row)
    
    table = wandb.Table(
        columns=["Model"] + [m.capitalize() for m in metrics],
        data=table_data
    )
    
    wandb.log({"model_comparison": table})
    
    # Create bar chart comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[model].get(metric, 0.0) for model in models]
        ax.bar(x + i * width, values, width, label=metric.capitalize())
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    wandb.log({"model_comparison_chart": wandb.Image(fig)})
    plt.close()
    
    wandb.finish()


# Example usage function
def example_usage():
    """Example of how to use WandbLogger"""
    
    # Initialize logger
    logger = WandbLogger(
        project_name="text-classification-imdb",
        experiment_name="bert_baseline",
        config={
            "model_type": "bert",
            "learning_rate": 2e-5,
            "batch_size": 16,
            "epochs": 10
        },
        tags=["bert", "baseline", "imdb"],
        notes="Baseline BERT model for sentiment analysis"
    )
    
    # Start logging
    logger.init_wandb("bert")
    
    # Log training metrics (example)
    for epoch in range(10):
        train_loss = 0.5 - epoch * 0.05  # Mock decreasing loss
        val_accuracy = 0.6 + epoch * 0.03  # Mock increasing accuracy
        
        logger.log_metrics({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/accuracy": val_accuracy,
            "learning_rate": 2e-5 * (0.9 ** epoch)
        }, step=epoch)
    
    # Log confusion matrix (example)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    logger.log_confusion_matrix(y_true, y_pred, title="Test Results")
    
    # Finish logging
    logger.finish()


if __name__ == "__main__":
    example_usage()
