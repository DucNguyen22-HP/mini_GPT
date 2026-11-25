import torch
from torch import nn
from torch.nn import functional as F


class GPTInputEmbedding(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

    def forward(self, token_ids):
        length = token_ids.shape[1]
        if not 1 <= length <= self.block_size:
            raise ValueError(f"Expected 1 to {self.block_size} input tokens per sequence")
        positions = torch.arange(length, device=token_ids.device)
        return self.token_embedding(token_ids) + self.position_embedding(positions)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = dropout
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.n_head, self.head_size)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        x = x.transpose(1, 2).reshape(batch, length, width)
        return self.output_dropout(self.projection(x))


class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_embd)
        self.attention = CausalSelfAttention(n_embd, n_head, dropout)
        self.norm2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout=0.1):
        super().__init__()
        if min(vocab_size, block_size, n_layer, n_head, n_embd) < 1 or n_embd % n_head:
            raise ValueError("Model sizes must be positive and n_embd must be divisible by n_head")
        self.block_size = block_size
        self.embedding = GPTInputEmbedding(vocab_size, block_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *(DecoderBlock(n_embd, n_head, dropout) for _ in range(n_layer))
        )
        self.norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._initialize)
        self.lm_head.weight = self.embedding.token_embedding.weight

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, token_ids):
        x = self.dropout(self.embedding(token_ids))
        x = self.blocks(x)
        return self.lm_head(self.norm(x))
