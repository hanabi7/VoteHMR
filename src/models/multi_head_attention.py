import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.matmul(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            attention = attention.masked_fill_(attn_mask, -np.inf)
            # mask_select
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.matmul(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_K = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_V = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_Q = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        residual = query
        batch_size = key.size(0)
        key = self.linear_K(key)
        value = self.linear_V(value)
        query = self.linear_Q(query)
        key = key.view(batch_size*self.num_heads, -1, self.dim_per_head)
        value = value.view(batch_size*self.num_heads, -1, self.dim_per_head)
        query = query.view(batch_size*self.num_heads, -1, self.dim_per_head)
        if attn_mask:
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1)
        scale = (key.size(-1) // self.num_heads) ** -0.5
        context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)
        context = context.view(batch_size, -1, self.dim_per_head * self.num_heads)
        output = self.linear_final(context)
        output = self.dropout(output)
        output = self.layer_norm(residual + output)
        return output, attention
