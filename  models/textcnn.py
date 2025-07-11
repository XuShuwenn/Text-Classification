import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """Implementation of TextCNN model"""
    def __init__(self,config):
        super(TextCNN,self).__init__()

        self.vocab_size=config['vocab_size']#词表大小
        self.embed_dim = config['embed_dim']#词嵌入维度
        self.num_filters = config['num_filters']#卷积核数量
        self.filter_sizes = config['filter_sizes']#卷积核大小列表
        self.num_classes = config['num_classes']#分类类别数
        self.dropout = config['dropout']#dropout

        #word embedding layer
        self.embedding=nn.Embedding(self.vocab_size,self.embed_dim)

        #Convolutional layer
        self.convs=nn.ModuleList([
            nn.Conv2d(1,self.nun_filters,(k,self.embed_dim))
            for k in self.filter_sizes
        ])     
        #Dropout layer
        self.dropout_layer=nn.Dropout(self.dropout)
        #MLP layer
        self.fc=nn.Linear(len(self.filter_sizes)*self.num_filters,self.num_classes)
    
    def forward(self, x):
        #x shape:(batch_size,seq_len)
        x=self.embedding(x)     # (batch_size, seq_len, embed_dim)
        x=x.unsqueeze(1)        # (batch_size, 1, seq_len, embed_dim)
        
        #卷积和池化层
        conv_results=[]
        for conv in self.convs:
            conv_out=F.relu(conv(x))
            # (batch_size, num_filters, conv_seq_len, 1)
            conv_out=conv_out.squeeze(3)
            # (batch_size, num_filters, conv_seq_len)
            pool_out=F.max_pool1d(conv_out,conv_out.size(2))
            # (batch_size, num_filters)
            conv_results.qppend(pool_out)

        #拼接所有卷积结果
        x=torch.cat(conv_results,1)
        #(batch_size, len(filter_sizes) * num_filters)
        
        # Dropout和Classification
        x=self.dropout_layer(x)
        x=self.fc(x)

        return x
    
