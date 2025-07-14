import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import os
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import wandb

from config import TRAINING_CONFIG, PATHS, DEVICE, MODEL_CONFIG
from utils.wandb_util import WandbLogger
from models import TextCNN, LSTM, BERTClassifier
from data_processor import get_data_loaders


def get_model(model_type, config=None):
    """
    Factory function to create model instances.
    
    Args:
        model_type (str): Type of model ('textcnn', 'lstm', 'bert')
        config (dict): Model configuration parameters
        
    Returns:
        nn.Module: Instantiated model
        
    Raises:
        ValueError: If model_type is not supported
    """
    model_registry = {
        'textcnn': TextCNN,
        'lstm': LSTM,
        'bert': BERTClassifier
    }
    
    if model_type not in model_registry:
        supported_models = list(model_registry.keys())
        raise ValueError(f"Unsupported model type: {model_type}. "
                        f"Supported models: {supported_models}")
    
    model_class = model_registry[model_type]
    
    if config is None:
        config = MODEL_CONFIG[model_type]
    
    return model_class(config)


class Trainer:
    """训练器类"""
    
    def __init__(self, model_type, model_config=None, use_wandb=True):
        self.model_type = model_type
        self.model_config = model_config
        self.device = DEVICE
        
        # 创建模型
        self.model = get_model(model_type, model_config).to(self.device)
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        
        # 训练状态
        self.best_accuracy = 0.0
        self.train_losses = []
        self.train_accuracies = []  # 添加训练准确率记录
        self.val_accuracies = []
        
        # Wandb logger
        self.wandb_logger = None
        if use_wandb:
            self.wandb_logger = WandbLogger(project_name="text-classification-imdb")
            self.wandb_logger.init_wandb(model_type)
            self.wandb_logger.log_model_architecture(self.model, (32, 512))
        
        print(f"Initialized trainer for {model_type}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_optimizer(self, train_loader):
        """设置优化器和学习率调度器"""
        # 根据模型类型设置不同的学习率
        if self.model_type == 'bert':
            lr = TRAINING_CONFIG['learning_rate']
        else:
            # 使用配置中的学习率
            from config import LEARNING_RATES
            lr = LEARNING_RATES.get(self.model_type, 0.001)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=TRAINING_CONFIG['weight_decay']
        )
        
        # 学习率调度器
        if self.model_type == 'bert':
            num_training_steps = len(train_loader) * TRAINING_CONFIG['epochs']
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=TRAINING_CONFIG['warmup_steps'],
                num_training_steps=num_training_steps
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=3, gamma=0.1
            )
    
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 数据移动到设备
            if self.model_type == 'bert':
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                outputs = self.model(input_ids, attention_mask)
            else:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                outputs = self.model(input_ids)
            
            # 计算损失
            loss = self.criterion(outputs, labels)
            
            # 计算准确率
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪 - 对LSTM使用更严格的裁剪
            if self.model_type == 'lstm':
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    0.5  # LSTM使用更小的梯度裁剪阈值
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    TRAINING_CONFIG['max_grad_norm']
                )
            
            # 更新参数
            self.optimizer.step()
            
            if self.scheduler and self.model_type == 'bert':
                self.scheduler.step()
            
            total_loss += loss.item()
            
            # 计算当前准确率
            current_accuracy = correct_predictions / total_predictions
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
                'acc': f'{current_accuracy:.4f}'
            })
            
            # 记录指标
            if self.wandb_logger and batch_idx % TRAINING_CONFIG['logging_steps'] == 0:
                global_step = epoch * num_batches + batch_idx
                self.wandb_logger.log_metrics({
                    'train/loss': loss.item(),
                    'train/avg_loss': total_loss / (batch_idx + 1),
                    'train/accuracy': current_accuracy,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }, step=global_step)
        
        if self.scheduler and self.model_type != 'bert':
            self.scheduler.step()
        
        avg_loss = total_loss / num_batches
        train_accuracy = correct_predictions / total_predictions
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(train_accuracy)
        
        return avg_loss, train_accuracy
    
    def evaluate(self, data_loader, split_name="val"):
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {split_name}"):
                # 数据移动到设备
                if self.model_type == 'bert':
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # 前向传播
                    outputs = self.model(input_ids, attention_mask)
                else:
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # 前向传播
                    outputs = self.model(input_ids)
                
                # 计算损失
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # 预测
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        avg_loss = total_loss / len(data_loader)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics, all_predictions, all_labels
    
    
    def log_results(self, predictions, labels, split_name="test"):
        """使用wandb记录结果"""
        if self.wandb_logger:
            self.wandb_logger.log_confusion_matrix(
                labels, predictions, 
                class_names=['Negative', 'Positive'],
                title=f'{self.model_type.upper()} - {split_name.title()}'
            )
            self.wandb_logger.log_classification_report(
                labels, predictions,
                class_names=['Negative', 'Positive'],
                prefix=f"{split_name}/"
            )
    
    def train(self, train_loader, val_loader):
        """完整的训练流程"""
        print(f"Starting training for {self.model_type}...")
        
        # 设置优化器
        self.setup_optimizer(train_loader)
        
        for epoch in range(TRAINING_CONFIG['epochs']):
            start_time = time.time()
            
            # 训练
            train_loss, train_accuracy = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_metrics, _, _ = self.evaluate(val_loader, "validation")
            val_accuracy = val_metrics['accuracy']
            self.val_accuracies.append(val_accuracy)
            
            # 记录训练和验证指标
            if self.wandb_logger:
                global_step = (epoch + 1) * len(train_loader)  # 确保step值单调递增
                self.wandb_logger.log_metrics({
                    'train/accuracy': train_accuracy,
                    'val/accuracy': val_accuracy,
                    'val/loss': val_metrics['loss'],
                    'val/precision': val_metrics['precision'],
                    'val/recall': val_metrics['recall'],
                    'val/f1': val_metrics['f1'],
                    'epoch': epoch
                }, step=global_step)
            
            # 更新最佳准确率
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1}/{TRAINING_CONFIG['epochs']} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                  f"Val Accuracy: {val_accuracy:.4f}, Time: {epoch_time:.2f}s")
        
        # 记录学习曲线并完成训练
        if self.wandb_logger:
            # 记录训练和验证准确率曲线
            self.wandb_logger.log_accuracy_curves(
                self.train_accuracies, 
                self.val_accuracies, 
                self.train_losses, 
                self.model_type
            )
            
            # 记录最终的学习曲线
            val_losses = [self.val_accuracies[i] * -1 for i in range(len(self.val_accuracies))]  # 模拟验证损失
            self.wandb_logger.log_learning_curves(self.train_losses, self.val_accuracies, val_losses)
        
        print(f"Training completed. Best accuracy: {self.best_accuracy:.4f}")
    
    def test(self, test_loader):
        """测试模型"""
        print(f"Testing {self.model_type}...")
        
        # 测试
        test_metrics, predictions, labels = self.evaluate(test_loader, "test")
        
        print(f"Test Results for {self.model_type.upper()}:")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall: {test_metrics['recall']:.4f}")
        print(f"F1-Score: {test_metrics['f1']:.4f}")
        
        # 记录测试结果
        if self.wandb_logger:
            self.wandb_logger.log_metrics({
                'test/accuracy': test_metrics['accuracy'],
                'test/precision': test_metrics['precision'],
                'test/recall': test_metrics['recall'],
                'test/f1': test_metrics['f1']
            })
            
        # 使用wandb记录可视化结果
        self.log_results(predictions, labels, "test")
        
        return test_metrics
    
def compare_all_models(results):
    """使用wandb对比所有模型"""
    from utils.wandb_util import compare_models_wandb
    
    print(f"\n{'='*50}")
    print("MODEL COMPARISON")
    print(f"{'='*50}")
    
    # 控制台输出对比
    for model_type, metrics in results.items():
        print(f"{model_type.upper()}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print()
    
    # 使用wandb创建对比可视化
    compare_models_wandb(results)


if __name__ == "__main__":
    """主函数 - 支持命令行参数训练指定模型"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Text Classification Training')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['textcnn', 'lstm', 'bert', 'all'],
                       help='Model to train: textcnn, lstm, bert, or all (default: all)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable wandb logging')
    
    args = parser.parse_args()
    
    # 根据参数确定要训练的模型
    if args.model == 'all':
        model_types = ['textcnn', 'lstm', 'bert']
    else:
        model_types = [args.model]
    
    # 应用命令行参数覆盖配置
    if args.epochs is not None:
        TRAINING_CONFIG['epochs'] = args.epochs
    if args.batch_size is not None:
        from config import DATA_CONFIG
        DATA_CONFIG['batch_size'] = args.batch_size
    if args.lr is not None:
        from config import LEARNING_RATES
        for model_type in model_types:
            if model_type in LEARNING_RATES:
                LEARNING_RATES[model_type] = args.lr
    
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*50}")
        
        # 获取数据加载器
        train_loader, val_loader, test_loader, processor = get_data_loaders(model_type)
        
        # 更新词汇表大小（对于非BERT模型）
        if model_type != 'bert' and hasattr(processor, 'vocab'):
            from config import MODEL_CONFIG
            MODEL_CONFIG[model_type]['vocab_size'] = len(processor.vocab)
        
        # 创建训练器
        trainer = Trainer(model_type, use_wandb=not args.no_wandb)
        
        # 训练
        trainer.train(train_loader, val_loader)  # 使用val_loader进行验证
        
        # 测试
        test_metrics = trainer.test(test_loader)
        results[model_type] = test_metrics
        
        # 完成当前模型的wandb记录
        if trainer.wandb_logger:
            trainer.wandb_logger.finish()
    
    # 如果训练了多个模型，进行对比
    if len(model_types) > 1:
        compare_all_models(results)
    else:
        # 只训练一个模型时，显示结果
        model_type = model_types[0]
        print(f"\n{model_type.upper()} Training Complete!")
        print(f"Final Test Accuracy: {results[model_type]['accuracy']:.4f}")
        print(f"Final Test F1-Score: {results[model_type]['f1']:.4f}")
    


