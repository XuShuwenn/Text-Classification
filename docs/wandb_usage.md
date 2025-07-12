# Wandb 使用指南

## 快速开始

### 1. 安装和登录

```bash
# 安装 wandb
pip install wandb

# 登录账户（首次使用）
wandb login
```

### 2. 基本使用

```python
from utils.wandb_util import WandbLogger

# 创建 logger
logger = WandbLogger(
    project_name="imdb-classification",
    experiment_name="bert_experiment",
    config={"model": "bert", "lr": 2e-5},
    tags=["bert", "baseline"]
)

# 初始化
logger.init_wandb("bert")

# 记录指标
logger.log_metrics({
    "train/loss": 0.5,
    "val/accuracy": 0.85
}, step=epoch)

# 结束实验
logger.finish()
```

## 核心功能

### 1. 训练指标记录

```python
# 在训练循环中
logger.log_metrics({
    "train/loss": loss.item(),
    "train/lr": optimizer.param_groups[0]['lr'],
    "epoch": epoch
}, step=global_step)
```

### 2. 混淆矩阵

```python
# 记录混淆矩阵
logger.log_confusion_matrix(
    y_true=true_labels,
    y_pred=predictions,
    title="Test Results"
)
```

### 3. 学习曲线

```python
# 记录学习曲线
logger.log_learning_curves(
    train_losses=[0.8, 0.6, 0.4],
    val_accuracies=[0.7, 0.8, 0.85]
)
```

### 4. 模型信息

```python
# 记录模型架构
logger.log_model_architecture(model, input_shape=(16, 512))

# 保存模型
logger.save_model_artifact(
    model_path="models/best_model.pt",
    model_name="bert_best"
)
```

## 集成到训练脚本

### 修改 Trainer 类

```python
class Trainer:
    def __init__(self, model_type, use_wandb=True):
        # ...existing code...
        
        if use_wandb:
            self.wandb_logger = WandbLogger(
                project_name="imdb-text-classification",
                experiment_name=f"{model_type}_exp",
                config={"model_type": model_type}
            )
            self.wandb_logger.init_wandb(model_type)
    
    def train_epoch(self, train_loader, epoch):
        # ...existing training code...
        
        # 记录训练指标
        if hasattr(self, 'wandb_logger'):
            self.wandb_logger.log_metrics({
                "train/loss": loss.item(),
                "train/lr": self.optimizer.param_groups[0]['lr']
            }, step=global_step)
    
    def evaluate(self, data_loader, split="val"):
        # ...existing evaluation code...
        
        # 记录验证指标
        if hasattr(self, 'wandb_logger'):
            self.wandb_logger.log_metrics({
                f"{split}/accuracy": accuracy,
                f"{split}/f1": f1_score
            })
            
            # 测试时记录混淆矩阵
            if split == "test":
                self.wandb_logger.log_confusion_matrix(
                    true_labels, predictions
                )
```

## 模型对比

```python
from utils.wandb_util import compare_models_wandb

# 比较多个模型结果
results = {
    "textcnn": {"accuracy": 0.87, "f1": 0.86},
    "lstm": {"accuracy": 0.85, "f1": 0.84}, 
    "bert": {"accuracy": 0.93, "f1": 0.92}
}

compare_models_wandb(results)
```

## 最佳实践

### 1. 命名规范
- 项目名：`imdb-text-classification`
- 实验名：`{model_type}_{description}`
- 标签：`[model_type, dataset, method]`

### 2. 记录频率
- 训练指标：每 100 步
- 验证指标：每个 epoch
- 可视化：仅在测试时

### 3. 配置示例

```python
configs = {
    "textcnn": {
        "model_type": "textcnn",
        "embedding_dim": 100,
        "num_filters": 100,
        "filter_sizes": [3, 4, 5],
        "dropout": 0.5
    },
    "lstm": {
        "model_type": "lstm", 
        "hidden_dim": 128,
        "num_layers": 2,
        "bidirectional": True
    },
    "bert": {
        "model_type": "bert",
        "model_name": "bert-base-uncased",
        "learning_rate": 2e-5
    }
}
```

## 常用指标

### 训练指标
- `train/loss` - 训练损失
- `train/lr` - 学习率
- `train/epoch` - 当前 epoch

### 验证指标  
- `val/accuracy` - 验证准确率
- `val/loss` - 验证损失
- `val/f1` - F1 分数

### 模型信息
- `model/total_parameters` - 总参数量
- `model/trainable_parameters` - 可训练参数

## 查看结果

登录 [wandb.ai](https://wandb.ai) 查看：
- 实时训练曲线
- 模型性能对比
- 混淆矩阵
- 超参数分析

## 故障排除

### 常见问题

1. **登录失败**
   ```bash
   wandb login --relogin
   ```

2. **网络问题**
   ```python
   # 离线模式
   os.environ["WANDB_MODE"] = "offline"
   ```

3. **禁用 wandb**
   ```python
   logger = WandbLogger()  # 不初始化
   # 或
   os.environ["WANDB_DISABLED"] = "true"
   ```

## 完整示例

```python
# 在 main() 函数中
def main():
    model_types = ['textcnn', 'lstm', 'bert']
    results = {}
    
    for model_type in model_types:
        # 创建带 wandb 的训练器
        trainer = Trainer(model_type, use_wandb=True)
        
        # 训练和测试
        trainer.train(train_loader, val_loader)
        test_metrics = trainer.test(test_loader)
        results[model_type] = test_metrics
    
    # 模型对比
    compare_models_wandb(results)
```

---

> 💡 **提示**: 第一次使用需要在 wandb.ai 注册账户并获取 API key
