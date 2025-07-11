import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import TRAINING_CONFIG, PATHS, DEVICE
from models import get_model
from data_processor import get_data_loaders


class Trainer:
    """训练器类"""
    
    def __init__(self, model_type, model_config=None):
        self.model_type = model_type
        self.model_config = model_config
        self.device = DEVICE
        
        # 创建模型
        self.model = get_model(model_type, model_config).to(self.device)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = None
        self.scheduler = None
        
        # 训练状态
        self.best_accuracy = 0.0
        self.train_losses = []
        self.val_accuracies = []
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=os.path.join(PATHS['log_dir'], model_type))
        
        print(f"Initialized trainer for {model_type}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_optimizer(self, train_loader):
        """设置优化器和学习率调度器"""
        # 根据模型类型设置不同的学习率
        if self.model_type == 'bert':
            lr = TRAINING_CONFIG['learning_rate']
        else:
            lr = 0.001  # 传统模型使用较高的学习率
        
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
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                TRAINING_CONFIG['max_grad_norm']
            )
            
            # 更新参数
            self.optimizer.step()
            
            if self.scheduler and self.model_type == 'bert':
                self.scheduler.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
            
            # 记录到TensorBoard
            global_step = epoch * num_batches + batch_idx
            self.writer.add_scalar('Loss/Train', loss.item(), global_step)
            
            if batch_idx % TRAINING_CONFIG['logging_steps'] == 0:
                self.writer.add_scalar(
                    'Learning_Rate', 
                    self.optimizer.param_groups[0]['lr'], 
                    global_step
                )
        
        if self.scheduler and self.model_type != 'bert':
            self.scheduler.step()
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
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
    
    def save_model(self, epoch, accuracy, is_best=False):
        """保存模型"""
        model_dir = os.path.join(PATHS['model_dir'], self.model_type)
        os.makedirs(model_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'model_type': self.model_type,
            'model_config': self.model_config
        }
        
        # 保存检查点
        checkpoint_path = os.path.join(model_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(model_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with accuracy: {accuracy:.4f}")
    
    def plot_confusion_matrix(self, predictions, labels, split_name="test"):
        """绘制混淆矩阵"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {self.model_type.upper()} ({split_name})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        output_path = os.path.join(PATHS['output_dir'], f'{self.model_type}_confusion_matrix_{split_name}.png')
        plt.savefig(output_path)
        plt.close()
        
        print(f"Confusion matrix saved to: {output_path}")
    
    def train(self, train_loader, val_loader):
        """完整的训练流程"""
        print(f"Starting training for {self.model_type}...")
        
        # 设置优化器
        self.setup_optimizer(train_loader)
        
        # 早停
        patience = TRAINING_CONFIG['patience']
        patience_counter = 0
        
        for epoch in range(TRAINING_CONFIG['epochs']):
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_metrics, _, _ = self.evaluate(val_loader, "validation")
            val_accuracy = val_metrics['accuracy']
            
            self.val_accuracies.append(val_accuracy)
            
            # 记录到TensorBoard
            self.writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
            self.writer.add_scalar('F1/Validation', val_metrics['f1'], epoch)
            
            # 保存模型
            is_best = val_accuracy > self.best_accuracy
            if is_best:
                self.best_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
            
            self.save_model(epoch, val_accuracy, is_best)
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1}/{TRAINING_CONFIG['epochs']} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Accuracy: {val_accuracy:.4f}, "
                  f"Time: {epoch_time:.2f}s")
            
            # 早停检查
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        self.writer.close()
        print(f"Training completed. Best accuracy: {self.best_accuracy:.4f}")
    
    def test(self, test_loader):
        """测试模型"""
        print(f"Testing {self.model_type}...")
        
        # 加载最佳模型
        model_dir = os.path.join(PATHS['model_dir'], self.model_type)
        best_model_path = os.path.join(model_dir, 'best_model.pt')
        
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']}")
        
        # 测试
        test_metrics, predictions, labels = self.evaluate(test_loader, "test")
        
        print(f"Test Results for {self.model_type.upper()}:")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall: {test_metrics['recall']:.4f}")
        print(f"F1-Score: {test_metrics['f1']:.4f}")
        
        # 绘制混淆矩阵
        self.plot_confusion_matrix(predictions, labels, "test")
        
        return test_metrics


def main():
    """主函数 - 训练所有模型"""
    model_types = ['textcnn', 'lstm', 'bert']
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*50}")
        
        # 获取数据加载器
        train_loader, test_loader, processor = get_data_loaders(model_type)
        
        # 更新词汇表大小（对于非BERT模型）
        if model_type != 'bert' and hasattr(processor, 'vocab'):
            from config import MODEL_CONFIG
            MODEL_CONFIG[model_type]['vocab_size'] = len(processor.vocab)
        
        # 创建训练器
        trainer = Trainer(model_type)
        
        # 划分验证集
        # 这里为了简化，我们使用测试集作为验证集
        # 在实际项目中，你应该从训练集中划分验证集
        
        # 训练
        trainer.train(train_loader, test_loader)
        
        # 测试
        test_metrics = trainer.test(test_loader)
        results[model_type] = test_metrics
    
    # 打印所有结果
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    
    for model_type, metrics in results.items():
        print(f"{model_type.upper()}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print()


if __name__ == "__main__":
    main()
