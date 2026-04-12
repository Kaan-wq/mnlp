import torch.nn as nn
import torch.nn.functional as F
from .config import GPTConfig
from .mha import MaskedMultiHeadSelfAttention
import torch


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 *
                             config.n_embd, device=config.device)
        self.fc2 = nn.Linear(
            4 * config.n_embd, config.n_embd, device=config.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, device=config.device)
        self.ln2 = nn.LayerNorm(config.n_embd, device=config.device)
        self.attn = MaskedMultiHeadSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
