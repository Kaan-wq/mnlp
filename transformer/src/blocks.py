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


def get_activation(config: GPTConfig) -> nn.Module:
    if config.activation_type == "gelu":
        return MLP(config)
    elif config.activation_type == "swiglu":
        return SwiGLU(config)
    else:
        raise ValueError(f"Unknown activation type: {config.activation_type}")


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.fc2._is_residual_proj = True  # mark for scaled init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class SwiGLU(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden_dim = int(8 / 3 * config.n_embd)
        self.fc1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.fc2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.fc3._is_residual_proj = True  # mark for scaled init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = get_norm(config.norm_type, config.n_embd)
        self.ln2 = get_norm(config.norm_type, config.n_embd)
        self.attn = MaskedGroupedQuerySelfAttention(config)
        self.mlp = get_activation(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, past_key_values=None, layer_idx=0, use_cache=False):
        attn_out = self.attn(
            self.ln1(x),
            past_key_values=past_key_values,
            layer_idx=layer_idx,
            use_cache=use_cache,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x
