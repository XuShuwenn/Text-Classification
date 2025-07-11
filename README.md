# Text-Classification

这是一个使用IMDB数据集进行文本分类的项目，实现了三种不同的深度学习模型进行情感分析。

## 数据集

使用Hugging Face上的Stanford IMDB数据集：[stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)

- 训练集：25,000条电影评论
- 测试集：25,000条电影评论
- 标签：0（负面）、1（正面）

## 模型架构

本项目实现了三种不同的模型：

### 1. TextCNN
- 使用多个不同大小的卷积核（3，4，5）
- 词嵌入维度：100
- 卷积核数量：100
- 包含Dropout层防止过拟合

### 2. LSTM
- 双向LSTM网络
- 词嵌入维度：100
- 隐藏层维度：128
- 层数：2层
- 包含Dropout层

### 3. BERT
- 使用预训练的bert-base-uncased模型
- 在BERT基础上添加分类头
- 支持fine-tuning

## 项目结构

```
Text-Classification/
├── main.py              # 主运行脚本
├── config.py            # 配置文件
├── data_processor.py    # 数据处理模块
├── models.py            # 模型定义
├── train.py            # 训练脚本
├── evaluate.py         # 评估脚本
├── inference.py        # 推理脚本
├── requirements.txt    # 依赖包
├── data/              # 数据目录
├── models/            # 保存的模型
├── output/            # 输出结果
├── logs/              # 训练日志
└── cache/             # 缓存目录
```

## 安装和使用

### 1. 环境设置

```bash
# 克隆项目
git clone <repository-url>
cd Text-Classification

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 快速开始

```bash
# 检查环境
python main.py check

# 设置项目目录
python main.py setup

# 训练所有模型
python main.py train

# 评估模型
python main.py evaluate

# 推理（交互模式）
python main.py infer --model bert --interactive

# 推理（单次）
python main.py infer --model bert --text "This movie is amazing!"
```

### 3. 详细使用

#### 训练模型
```bash
# 训练所有模型（TextCNN、LSTM、BERT）
python train.py

# 或使用主脚本
python main.py train
```

#### 评估模型
```bash
# 评估所有训练好的模型
python evaluate.py

# 或使用主脚本
python main.py evaluate
```

#### 推理
```bash
# 使用BERT模型进行推理
python inference.py --model bert --text "Great movie!"

# 交互模式
python inference.py --model bert --interactive

# 使用其他模型
python inference.py --model textcnn --text "Terrible film!"
python inference.py --model lstm --text "Amazing story!"
```

## 配置说明

主要配置在`config.py`中：

- `DATA_CONFIG`: 数据相关配置（批次大小、序列长度等）
- `MODEL_CONFIG`: 模型相关配置（词汇表大小、嵌入维度等）
- `TRAINING_CONFIG`: 训练相关配置（学习率、epochs等）

## 输出结果

训练和评估完成后，会生成以下文件：

- `models/`: 保存的模型检查点
- `output/model_comparison.png`: 模型性能对比图
- `output/all_confusion_matrices.png`: 混淆矩阵
- `output/evaluation_report.txt`: 详细评估报告
- `logs/`: TensorBoard日志文件

## 模型性能

典型的性能表现（仅供参考）：

| 模型 | 准确率 | F1分数 | 参数量 |
|------|--------|--------|--------|
| TextCNN | ~87% | ~87% | ~2.5M |
| LSTM | ~85% | ~85% | ~3.2M |
| BERT | ~93% | ~93% | ~110M |

## 技术特性

- **多模型支持**: 三种不同架构的模型
- **完整流程**: 数据处理、训练、评估、推理一条龙
- **可视化**: 训练过程监控和结果可视化
- **模块化设计**: 代码结构清晰，易于扩展
- **配置化**: 所有超参数可配置
- **错误分析**: 详细的模型错误分析
- **早停机制**: 防止过拟合
- **TensorBoard**: 训练过程可视化

## 扩展建议

1. **数据增强**: 添加数据增强技术提升模型泛化能力
2. **集成学习**: 将多个模型结果进行集成
3. **超参数优化**: 使用Optuna等工具进行自动超参数搜索
4. **模型量化**: 对BERT模型进行量化减少部署成本
5. **多语言支持**: 扩展到其他语言的情感分析

## 依赖包

主要依赖：
- PyTorch: 深度学习框架
- Transformers: BERT模型
- Datasets: Hugging Face数据集库
- Scikit-learn: 机器学习工具
- Matplotlib/Seaborn: 可视化

完整依赖列表见`requirements.txt`。


