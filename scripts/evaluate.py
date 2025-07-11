import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os

from config import PATHS, DEVICE
from models import get_model
from data_processor import get_data_loaders
from train import Trainer


def load_model(model_type, model_path=None):
    """加载训练好的模型"""
    if model_path is None:
        model_dir = os.path.join(PATHS['model_dir'], model_type)
        model_path = os.path.join(model_dir, 'best_model.pt')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # 创建模型
    model_config = checkpoint.get('model_config')
    model = get_model(model_type, model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    print(f"Loaded {model_type} model from {model_path}")
    print(f"Model was trained for {checkpoint['epoch']} epochs")
    print(f"Best validation accuracy: {checkpoint['accuracy']:.4f}")
    
    return model


def evaluate_model(model, model_type, test_loader):
    """评估单个模型"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            if model_type == 'bert':
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                
                outputs = model(input_ids, attention_mask)
            else:
                input_ids = batch['input_ids'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                
                outputs = model(input_ids)
            
            # 获取概率
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)


def plot_model_comparison(results_dict):
    """绘制模型比较图"""
    models = list(results_dict.keys())
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # 准备数据
    data = []
    for model in models:
        metrics_values = results_dict[model]
        data.append([
            metrics_values['accuracy'],
            metrics_values['precision'],
            metrics_values['recall'],
            metrics_values['f1']
        ])
    
    # 创建DataFrame
    df = pd.DataFrame(data, columns=metrics, index=[m.upper() for m in models])
    
    # 绘制条形图
    fig, ax = plt.subplots(figsize=(12, 8))
    df.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Model Performance Comparison', fontsize=16)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Models', fontsize=12)
    ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 1)
    
    # 添加数值标签
    for i, model in enumerate(models):
        for j, metric in enumerate(metrics):
            value = data[i][j]
            ax.text(i + j*0.25 - 0.375, value + 0.01, f'{value:.3f}', 
                   ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(PATHS['output_dir'], 'model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model comparison plot saved to: {output_path}")


def plot_confusion_matrices(results_dict):
    """绘制所有模型的混淆矩阵"""
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for i, (model_type, result) in enumerate(results_dict.items()):
        predictions = result['predictions']
        labels = result['labels']
        
        cm = confusion_matrix(labels, predictions)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   ax=axes[i])
        axes[i].set_title(f'{model_type.upper()}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(PATHS['output_dir'], 'all_confusion_matrices.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"All confusion matrices saved to: {output_path}")


def analyze_errors(predictions, labels, probabilities, model_type, test_texts=None):
    """分析模型错误"""
    # 找出错误预测的样本
    errors = predictions != labels
    error_indices = np.where(errors)[0]
    
    print(f"\n{model_type.upper()} Error Analysis:")
    print(f"Total errors: {len(error_indices)} / {len(labels)} ({len(error_indices)/len(labels)*100:.2f}%)")
    
    if len(error_indices) > 0:
        # 分析错误类型
        false_positives = np.where((predictions == 1) & (labels == 0))[0]
        false_negatives = np.where((predictions == 0) & (labels == 1))[0]
        
        print(f"False Positives: {len(false_positives)}")
        print(f"False Negatives: {len(false_negatives)}")
        
        # 分析置信度
        error_probs = probabilities[error_indices]
        error_confidences = np.max(error_probs, axis=1)
        
        print(f"Average confidence on errors: {error_confidences.mean():.3f}")
        print(f"Min confidence on errors: {error_confidences.min():.3f}")
        print(f"Max confidence on errors: {error_confidences.max():.3f}")
        
        # 低置信度错误
        low_conf_threshold = 0.6
        low_conf_errors = error_indices[error_confidences < low_conf_threshold]
        print(f"Low confidence errors (<{low_conf_threshold}): {len(low_conf_errors)}")


def generate_detailed_report(results_dict):
    """生成详细的评估报告"""
    report_path = os.path.join(PATHS['output_dir'], 'evaluation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("IMDB Text Classification - Model Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        
        # 模型性能对比
        f.write("Model Performance Summary:\n")
        f.write("-" * 30 + "\n")
        
        for model_type, metrics in results_dict.items():
            f.write(f"\n{model_type.upper()}:\n")
            f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1']:.4f}\n")
        
        # 最佳模型
        best_model = max(results_dict.keys(), key=lambda x: results_dict[x]['accuracy'])
        f.write(f"\nBest Model: {best_model.upper()}\n")
        f.write(f"Best Accuracy: {results_dict[best_model]['accuracy']:.4f}\n\n")
        
        # 详细分类报告
        f.write("Detailed Classification Reports:\n")
        f.write("-" * 35 + "\n")
        
        for model_type, result in results_dict.items():
            f.write(f"\n{model_type.upper()}:\n")
            report = classification_report(
                result['labels'], 
                result['predictions'],
                target_names=['Negative', 'Positive']
            )
            f.write(report)
            f.write("\n")
    
    print(f"Detailed evaluation report saved to: {report_path}")


def main():
    """主评估函数"""
    model_types = ['textcnn', 'lstm', 'bert']
    results = {}
    
    print("Starting model evaluation...")
    
    for model_type in model_types:
        print(f"\nEvaluating {model_type.upper()}...")
        
        try:
            # 加载数据
            _, test_loader, _ = get_data_loaders(model_type)
            
            # 加载模型
            model = load_model(model_type)
            
            # 评估
            predictions, labels, probabilities = evaluate_model(model, model_type, test_loader)
            
            # 计算指标
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            
            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='weighted'
            )
            
            results[model_type] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': predictions,
                'labels': labels,
                'probabilities': probabilities
            }
            
            # 错误分析
            analyze_errors(predictions, labels, probabilities, model_type)
            
            print(f"{model_type.upper()} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
        except Exception as e:
            print(f"Error evaluating {model_type}: {str(e)}")
            continue
    
    if results:
        # 生成可视化
        plot_model_comparison(results)
        plot_confusion_matrices(results)
        
        # 生成报告
        generate_detailed_report(results)
        
        print("\nEvaluation completed! Check the output directory for results.")
    else:
        print("No models were successfully evaluated.")


if __name__ == "__main__":
    main()
