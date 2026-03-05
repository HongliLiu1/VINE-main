import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskMultiHeadAttention(nn.Module):
    # 支持掩码操作的多头注意力机制
    def __init__(self,
                 head_num,
                 in_features,
                 bias=True,
                 dropout=0.0,
                 activation=F.relu):
        super(MaskMultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.attn_dropout = nn.Dropout(dropout)
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)
        
    def forward(self, q, k, v, mask=None, cross=False, add_attn=False):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            # 如果提供了 mask，它会重复扩展以适应每个头的计算（即重复 head_num 次）。
            mask = mask.repeat_interleave(self.head_num, 0)
        # 	计算 scaled dot-product 注意力，q, k, v 通过 scaled_dotproduct 方法进行处理，得到输出 y 和注意力权重 weights。
        y, weights = self.scaled_dotproduct(q, k, v, mask=mask, add_attn=add_attn)
        y = self._reshape_from_batches(y)
        y = self.linear_o(y)
        return y, weights
    
    @staticmethod
    def gen_history_mask(x):
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        # 对 q, k, v 进行重塑，使它们适应多头注意力机制的计算。此时每个注意力头的输入特征是 in_features / head_num。
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        # 将输出 x 从多头注意力的形状转换回原始形状。
        # 通过逆操作将 batch_size * head_num 转换回原始的 batch_size，并将特征维度恢复到原始大小。
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )
        
    def scaled_dotproduct(self, query, key, value, mask=None, cross_att=False, add_attn=False, tmp=0.1):
        dk = query.size()[-1]
        query, key = F.normalize(query, dim=2), F.normalize(key, dim=2)
        
        scores = query.matmul(key.transpose(-2, -1))
        # 果提供了 mask，它会用来调整 scores。
        # 如果 add_attn=True，则直接对分数进行乘法操作，
        # 否则使用 masked_fill 将掩码值设置为负无穷（-1e9），使得这些位置的注意力权重变为 0。
        if mask is not None:
            if add_attn:
                scores = scores * mask
            else:
                scores = scores.masked_fill(mask == 0, -1e9)
            
        attention = F.softmax(scores, dim=-1)
        
        attention = self.attn_dropout(attention)
        
        weight = attention.reshape(attention.size(0) // self.head_num, self.head_num, attention.size(1),attention.size(2) )
        weight = weight.mean(1)
        # 使用注意力权重加权求值
        return attention.matmul(value), weight