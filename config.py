# 配置文件
import os

# 数据相关配置
DATA_CONFIG = {
    'dataset_name': 'stanfordnlp/imdb',
    'max_length': 512,
    'batch_size': 16,
    'num_workers': 4,
    'test_size': 0.2,
    'random_state': 42
}

# 模型相关配置
MODEL_CONFIG = {
    'textcnn': {
        'vocab_size': 25000,
        'embed_dim': 100,
        'num_filters': 100,
        'filter_sizes': [3, 4, 5],
        'dropout': 0.5,
        'num_classes': 2
    },
    'lstm': {
        'vocab_size': 25000,
        'embed_dim': 100,
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.5,
        'num_classes': 2,
        'bidirectional': True
    },
    'bert': {
        'model_name': 'bert-base-uncased',
        'num_classes': 2,
        'dropout': 0.1
    }
}

# 训练相关配置
TRAINING_CONFIG = {
    'epochs': 10,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'warmup_steps': 1000,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'save_steps': 500,
    'eval_steps': 500,
    'logging_steps': 100,
    'patience': 3  # for early stopping
}

# 路径配置
PATHS = {
    'data_dir': './data',
    'model_dir': './models',
    'output_dir': './output',
    'log_dir': './logs',
    'cache_dir': './cache'
}

# 创建必要的目录
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)

# 设备配置
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
