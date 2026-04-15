import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig
from .gqa import MaskedGroupedQuerySelfAttention


def get_norm(norm_type: str, dim: int) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    elif norm_type == "rmsnorm":
        return nn.RMSNorm(dim)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.fc2._is_residual_proj = True  # mark for scaled init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = get_norm(config.norm_type, config.n_embd)
        self.ln2 = get_norm(config.norm_type, config.n_embd)
        self.attn = MaskedGroupedQuerySelfAttention(config)
        self.mlp = MLP(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x
