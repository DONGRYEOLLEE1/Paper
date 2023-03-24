import torch
import torch.nn as nn
from TransformerBlock import *
from SelfAttention import *


class Encoder(nn.Module):
    """
    positional encoding에 문장의 최대길이에 영향을 받기 때문에 max_length 설정
    max_length 설정값은 데이터셋에 따라 값이 달라질 수 있음
    """
    def __init__(self, 
                 src_vocab_size, 
                 embed_size, 
                 num_layers, 
                 heads, 
                 device, 
                 forward_expansion, 
                 dropout, 
                 max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.positional_encoding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout = dropout,
                    forward_expansion = forward_expansion
                )
            for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        
        out = self.dropout(self.word_embedding(x) + self.positional_encoding(x))
        
        for layer in self.layers:
            out = layer(out, out, out, mask)
            
        return out