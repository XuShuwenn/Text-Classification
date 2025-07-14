import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Optional


class WandbLogger:
    """Weights & Biases logger for text classification experiments"""
    
    def __init__(self, project_name: str = "text-classification-imdb"):
        """Initialize wandb logger with only project name"""
        self.project_name = project_name
        self.run = None
        
    def init_wandb(self, model_type: str):
        """Initialize wandb run"""
        self.run = wandb.init(
            project=self.project_name,
            name=model_type
        )
        
        print(f"🚀 Started wandb run: {self.run.name}")
        print(f"📊 View at: {self.run.url}")
        
    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        """Log metrics to wandb"""
        if self.run:
            wandb.log(metrics, step=step)
    
    def log_model_architecture(self, model: torch.nn.Module, input_shape: tuple):
        """Log model architecture and parameters"""
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
    
    def log_confusion_matrix(self, 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray, 
                           class_names: List[str] = None,
                           title: str = "Confusion Matrix"):
        """Log confusion matrix to wandb"""
        if not self.run:
            return
            
        if class_names is None:
            class_names = ["Negative", "Positive"]
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        # Log to wandb
        wandb.log({f"confusion_matrix/{title.lower().replace(' ', '_')}": wandb.Image(fig)}, step=wandb.run.step)
        plt.close(fig)
    
    def log_classification_report(self, 
                                y_true: np.ndarray, 
                                y_pred: np.ndarray,
                                class_names: List[str] = None,
                                prefix: str = ""):
        """Log detailed classification report"""
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
        """Log learning curves"""
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
        wandb.log({"learning_curves": wandb.Image(fig)}, step=wandb.run.step)
        plt.close()
    
    def log_accuracy_curves(self, 
                           train_accuracies: List[float],
                           val_accuracies: List[float],
                           train_losses: List[float],
                           model_type: str):
        """Log training and validation accuracy curves with training loss"""
        if not self.run:
            return
        
        epochs = range(1, len(train_accuracies) + 1)
        
        # Create accuracy comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Training and validation accuracy
        ax1.plot(epochs, train_accuracies, label='Train Accuracy', color='blue', marker='o')
        ax1.plot(epochs, val_accuracies, label='Val Accuracy', color='red', marker='s')
        ax1.set_title(f'{model_type.upper()} - Training and Validation Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        ax1.set_ylim(0, 1)
        
        # Training loss
        ax2.plot(epochs, train_losses, label='Train Loss', color='orange', marker='o')
        ax2.set_title(f'{model_type.upper()} - Training Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        wandb.log({f"{model_type}_accuracy_curves": wandb.Image(fig)}, step=wandb.run.step)
        plt.close(fig)
    
    def finish(self):
        """Finish wandb run"""
        if self.run:
            wandb.finish()
            print("🏁 Finished wandb run")


def compare_models_wandb(results: Dict[str, Dict], 
                        project_name: str = "text-classification-comparison"):
    """Create a comparison visualization of multiple models"""
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
    
    wandb.log({"model_comparison": table}, step=0)
    
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
    
    wandb.log({"model_comparison_chart": wandb.Image(fig)}, step=1)
    plt.close()
    
    wandb.finish()
