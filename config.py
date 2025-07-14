# 配置文件
import os

# 数据相关配置 - 针对IMDB数据集优化
DATA_CONFIG = {
    'dataset_name': 'stanfordnlp/imdb',
    'dataset_config': 'plain_text',
    'max_length': 512,  # IMDB评论平均长度较长，保持512
    'batch_size': 256,  # RTX A6000有48GB显存，可以支持更大的批次
    'num_workers': 16,  # 增加worker数量，充分利用多核CPU
    'test_size': 0.1,  # 从测试集中分出10%作为验证集
    'random_state': 42,
    'pin_memory': True,
    'persistent_workers': True,
    'vocab_size': 30000,  # IMDB数据集词汇量适中
    'min_freq': 2,  # 最小词频，过滤低频词
    'max_vocab_size': 50000  # 最大词汇表大小
}

# 模型相关配置 - 针对IMDB情感分析优化
MODEL_CONFIG = {
    'textcnn': {
        'vocab_size': 30000,
        'embed_dim': 512,  # 增加嵌入维度，充分利用显存
        'num_filters': 512,  # 增加卷积核数量
        'filter_sizes': [3, 4, 5],
        'dropout': 0.3,
        'num_classes': 2,
        'use_batch_norm': True
    },
    'lstm': {
        'vocab_size': 30000,
        'embed_dim': 64,  # 降低嵌入维度，减少过拟合
        'hidden_dim': 256,  # 降低隐藏层维度
        'num_layers': 2,  # 减少层数，避免梯度消失
        'dropout': 0.2,  # 降低dropout
        'num_classes': 2,
        'bidirectional': True,
    },
    'bert': {
        'model_name': 'bert-base-uncased',
        'num_classes': 2,
        'dropout': 0.3,
        'use_fp16': True,  # 启用混合精度，节省显存
        'freeze_bert': False,
        'bert_layers_to_freeze': 0,
        'classifier_dropout': 0.1
    }
}

# 训练相关配置 - 针对IMDB数据集优化
TRAINING_CONFIG = {
    'epochs': 20,  # 增加训练轮数，充分利用GPU
    'learning_rate': 2e-5,  #标准学习率
    'weight_decay': 0.01,
    'warmup_steps': 1000,
    'gradient_accumulation_steps': 2,  # 减少梯度累积，因为批次已经很大
    'max_grad_norm': 1.0,
    'save_steps': 500,
    'eval_steps': 500,
    'logging_steps': 100,
    'use_amp': True,  # 启用自动混合精度
    'scheduler': 'linear',
    'optimizer': 'adamw',
    'eval_strategy': 'steps',
    'save_strategy': 'steps',
    'load_best_model_at_end': True,
    'metric_for_best_model': 'accuracy',
    'greater_is_better': True
}

# 针对不同模型的学习率
LEARNING_RATES = {
    'textcnn': 4e-3,  # 增加学习率，因为批次更大
    'lstm': 1e-3,     # 降低LSTM学习率，避免训练不稳定
    'bert': 5e-5      # 稍微增加BERT学习率
}

# 路径配置
PATHS = {
    'model_dir': './models',
    'cache_dir': './cache',
    'results_dir': './results'  # 添加结果目录
}

# 创建必要的目录
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)

# 设备配置
import torch

# 可以通过环境变量指定GPU
cuda_device = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
if cuda_device.isdigit():
    DEVICE = torch.device(f'cuda:{cuda_device}')
else:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GPU内存优化配置
if torch.cuda.is_available():
    # 设置GPU内存分配策略
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # 获取GPU信息
    current_device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_device)
    gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3  # GB
    print(f"Using device: {DEVICE}")
    print(f"GPU: {gpu_name}")
    print(f"GPU Memory: {gpu_memory:.1f} GB")
    print(f"GPU ID: {current_device}")
    
    # 根据GPU内存和模型类型调整批次大小
    if gpu_memory >= 40:  # RTX A6000有48GB
        if 'bert' in MODEL_CONFIG:
            # BERT模型使用较小批次大小
            DATA_CONFIG['batch_size'] = 16
            TRAINING_CONFIG['gradient_accumulation_steps'] = 8  # 有效批次大小=16*8=128
        else:
            # 传统模型可以使用较大批次大小，但LSTM需要更小的批次
            DATA_CONFIG['batch_size'] = 32  # 降低批次大小
            TRAINING_CONFIG['gradient_accumulation_steps'] = 4  # 有效批次大小=32*4=128
        print("Using optimized batch size for RTX A6000")
    elif gpu_memory >= 20:
        DATA_CONFIG['batch_size'] = 32
        TRAINING_CONFIG['gradient_accumulation_steps'] = 4
        print("Using medium batch size for mid-range GPU")
    else:
        DATA_CONFIG['batch_size'] = 16
        TRAINING_CONFIG['gradient_accumulation_steps'] = 8
        print("Using small batch size for low-end GPU")
else:
    print(f"Using device: {DEVICE}")
    # CPU训练时减小批次大小
    DATA_CONFIG['batch_size'] = 8
    DATA_CONFIG['num_workers'] = 4
    TRAINING_CONFIG['gradient_accumulation_steps'] = 16

# 数据增强配置
AUGMENTATION_CONFIG = {
    'use_augmentation': True,
    'augmentation_methods': ['synonym_replacement', 'random_insertion', 'random_swap'],
    'augmentation_ratio': 0.1,  # 10%的数据进行增强
    'max_augmentations': 1  # 每个样本最多增强1次
}

# 评估配置
EVALUATION_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
    'save_predictions': True,
    'save_confusion_matrix': True,
    'save_classification_report': True,
    'confidence_threshold': 0.5
}

# 日志配置
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'training.log',
    'use_tensorboard': True,
    'use_wandb': True
}
