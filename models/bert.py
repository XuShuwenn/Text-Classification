import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class BERTClassifier(nn.Module):
    """BERT分类器"""
    
    def __init__(self, config):
        super(BERTClassifier, self).__init__()
        
        self.model_name = config['model_name']
        self.num_classes = config['num_classes']
        self.dropout = config['dropout']
        
        # 加载预训练BERT模型
        self.bert = AutoModel.from_pretrained(self.model_name)
        
        # 冻结BERT参数（可选）
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        
        # Dropout层
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # 分类头
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_classes)
    
    def forward(self, input_ids, attention_mask=None):
        # BERT前向传播
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 使用[CLS]标记的输出
        pooled_output = outputs.pooler_output
        
        # Dropout和分类
        output = self.dropout_layer(pooled_output)
        output = self.classifier(output)
        
        return output
    


