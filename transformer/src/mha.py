import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


class MaskedMultiHeadSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        self.config = config

        # Q, K, V projection matrices
        self.queries = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.keys = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.values = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.register_buffer(
            "att_mask",
            torch.tril(
                torch.ones(config.max_seq_length, config.max_seq_length)
            ).reshape(1, 1, config.max_seq_length, config.max_seq_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.size()
        h = self.config.n_head
        k = D // h

        # Q, K, V = (B, T, D) -> (B, T, h, k) -> (B, h, T, k)
        Q = self.queries(x).reshape(B, T, h, k).transpose(1, 2)
        K = self.keys(x).reshape(B, T, h, k).transpose(1, 2)
        V = self.values(x).reshape(B, T, h, k).transpose(1, 2)

        # Softmax along the 4th dim
        A_masked = ((Q @ K.transpose(2, 3)) / (k**0.5)).masked_fill(
            self.att_mask[:, :, :T, :T] == 0, float("-inf")
        )
        A = F.softmax(A_masked, dim=-1)

        # (B, h, T, k) -> (B, T, h, k) -> (B, T, D)
        Y = (A @ V).transpose(1, 2).reshape(B, T, D)
        output = self.out_proj(Y)

        return output
