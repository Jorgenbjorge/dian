import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义多头注意力类
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension must be divisible by num_heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_dim)

    def forward(self, values, keys, queries, mask=None):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split the embeddings into self.num_heads different pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.num_heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Attention score
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention_weights = F.softmax(energy / (self.embed_dim ** (1 / 2)), dim=3)

        # Apply the attention weights
        out = torch.einsum("nhql,nlhd->nqhd", [attention_weights, values]).reshape(
            N, query_len, self.num_heads * self.head_dim
        )

        # Pass through a final linear layer
        out = self.fc_out(out)
        return out

# 示例
embed_dim = 256
num_heads = 8
batch_size = 1
seq_len = 10

# 生成随机数据
values = torch.randn(batch_size, seq_len, embed_dim)
keys = torch.randn(batch_size, seq_len, embed_dim)
queries = torch.randn(batch_size, seq_len, embed_dim)

# 创建多头注意力层
attention_layer = MultiHeadAttention(embed_dim, num_heads)

# 得到权重和输出
output = attention_layer(values, keys, queries)

# 预期的shape:(batch_size, seq_len, embed_dim)
print(output.shape)