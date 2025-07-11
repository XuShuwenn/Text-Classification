import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import re
from tqdm import tqdm
from config import DATA_CONFIG, PATHS, DEVICE


class IMDBDataProcessor:
    """IMDB数据集处理器"""
    
    def __init__(self):
        self.tokenizer = None
        self.vocab = None
        self.word2idx = None
        self.idx2word = None
        
    def load_data(self):
        """加载IMDB数据集"""
        print("Loading IMDB dataset...")
        dataset = load_dataset(DATA_CONFIG['dataset_name'])
        
        # 转换为pandas DataFrame
        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        
        print(f"Train set size: {len(train_df)}")
        print(f"Test set size: {len(test_df)}")
        
        return train_df, test_df
    
    def preprocess_text(self, text):
        """文本预处理"""
        # 转换为小写
        text = text.lower()
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 移除特殊字符，保留字母、数字和空格
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def build_vocab(self, texts, vocab_size=25000):
        """构建词汇表"""
        print("Building vocabulary...")
        
        # 预处理文本
        processed_texts = [self.preprocess_text(text) for text in tqdm(texts)]
        
        # 统计词频
        word_counts = Counter()
        for text in processed_texts:
            words = text.split()
            word_counts.update(words)
        
        # 构建词汇表
        most_common = word_counts.most_common(vocab_size - 2)  # 保留<UNK>和<PAD>
        
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        
        for i, (word, _) in enumerate(most_common):
            self.word2idx[word] = i + 2
            self.idx2word[i + 2] = word
        
        self.vocab = self.word2idx
        print(f"Vocabulary size: {len(self.vocab)}")
        
        return self.vocab
    
    def text_to_sequence(self, text, max_length=512):
        """将文本转换为序列"""
        processed_text = self.preprocess_text(text)
        words = processed_text.split()[:max_length]
        
        sequence = []
        for word in words:
            if word in self.word2idx:
                sequence.append(self.word2idx[word])
            else:
                sequence.append(self.word2idx['<UNK>'])
        
        # 填充或截断
        if len(sequence) < max_length:
            sequence.extend([self.word2idx['<PAD>']] * (max_length - len(sequence)))
        else:
            sequence = sequence[:max_length]
        
        return sequence


class IMDBDataset(Dataset):
    """IMDB数据集类"""
    
    def __init__(self, texts, labels, processor, max_length=512, for_bert=False):
        self.texts = texts
        self.labels = labels
        self.processor = processor
        self.max_length = max_length
        self.for_bert = for_bert
        
        if for_bert:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        if self.for_bert:
            # BERT tokenization
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        else:
            # 传统tokenization
            sequence = self.processor.text_to_sequence(text, self.max_length)
            return {
                'input_ids': torch.tensor(sequence, dtype=torch.long),
                'labels': torch.tensor(label, dtype=torch.long)
            }


def get_data_loaders(model_type='textcnn'):
    """获取数据加载器"""
    processor = IMDBDataProcessor()
    
    # 加载数据
    train_df, test_df = processor.load_data()
    
    # 构建词汇表（仅对非BERT模型）
    if model_type != 'bert':
        processor.build_vocab(
            train_df['text'].tolist(), 
            vocab_size=DATA_CONFIG.get('vocab_size', 25000)
        )
    
    # 创建数据集
    for_bert = (model_type == 'bert')
    
    train_dataset = IMDBDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        processor,
        max_length=DATA_CONFIG['max_length'],
        for_bert=for_bert
    )
    
    test_dataset = IMDBDataset(
        test_df['text'].tolist(),
        test_df['label'].tolist(),
        processor,
        max_length=DATA_CONFIG['max_length'],
        for_bert=for_bert
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=DATA_CONFIG['batch_size'],
        shuffle=True,
        num_workers=DATA_CONFIG['num_workers']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=DATA_CONFIG['batch_size'],
        shuffle=False,
        num_workers=DATA_CONFIG['num_workers']
    )
    
    return train_loader, test_loader, processor


if __name__ == "__main__":
    # 测试数据处理
    train_loader, test_loader, processor = get_data_loaders('textcnn')
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # 查看一个batch的数据
    for batch in train_loader:
        print("Batch keys:", batch.keys())
        print("Input shape:", batch['input_ids'].shape)
        print("Labels shape:", batch['labels'].shape)
        break
