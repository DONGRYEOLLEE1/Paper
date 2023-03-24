import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """ 
    heads : n개의 파트로 나눌것인지
    """
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), 'Embbed size needs to be div by heads'
        
        self.values = nn.Linear(embed_size, embed_size, bias = False)
        self.keys = nn.Linear(embed_size, embed_size, bias = False)
        self.queries = nn.Linear(embed_size, embed_size, bias = False)
        
        self.fc_out = nn.Linear(embed_size, embed_size)  ## embed_size로 shape 맞춰주기
        
    def forward(self, values, keys, query, mask):
        
        N = query.shape[0]
        # target, source 문장과 항상 상응
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split embedding into self.heads pieces
        # 

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        

        energy = torch.einsum("nqhd,nkhd -> nhqk", [queries, keys]) 
        # queries shape : (N, query_len, heads, head_dim)
        # keys shape : (N, key_len, heads, head_dim)
        # energy shape : (N, heads, query_len, key_len) >> multiplication
        ## query_len = target, source sentence of length
        ## key_len = source sentence of length
        # how much pay attention to each word in input sentence
        
        if mask is not None:
            # Masking '-inf', and numercial values will be overflowed
            # mask matrix is same shape with K, Q and V
            energy = energy.masked_fill(mask == 0, float('-1e20'))
            
        # Attention(Q, K, V)
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim = 3) # normalizing
        
        out = torch.einsum("nhql,nlhd -> nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape : (N, heads, query_len, key_len)
        # values shape :  (N, value_len, heads, head_dim)
        # out shape : (N, query_len, heads, head_dim)  Encoder딴에선 K, V 길이는 항상 같음
        
        out = self.fc_out(out)
        
        return out