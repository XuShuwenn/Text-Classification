import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    """Implementation of LSTM model"""

    def __init__(self,config):
        super(LSTM,self).__init__()

        self.vocab_size=config['vocab_size']
        self.embed_dim = config['embed_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.num_classes = config['num_classes']
        self.dropout = config['dropout']
        self.bidirectional = config['bidirectional']

        #word embedding layer
        self.embedding=nn.Embedding(self.vocab_size,self.embed_dim)
        #LSTM layer
        self.lstm=nn.LSTM(
            self.embed_dim,
            self.hidden_dim,
            self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers>1 else 0,
            bidirectional=self.bidirectional
        )

        #计算LSTM的输出维度
        lstm_output_dim=self.hidden_dim*2 if self.bidirectional else self.hidden_dim
        #dropout layer
        self.dropout_layer=nn.Dropout(self.dropout)

        #MLP Layer
        self.fc=nn.Linear(lstm_output_dim,self.num_classes)


    def forward(self,x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_dim * num_directions)
        
        # 使用最后一个时间步的输出
        if self.bidirectional:
            # 拼接前向和后向的最后隐藏状态
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # Dropout和分类
        output = self.dropout_layer(hidden)
        output = self.fc(output)
        
        return output
       

