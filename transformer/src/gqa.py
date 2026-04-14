import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


class MaskedGroupedQuerySelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_head % config.n_kv_head == 0, (
            f"n_head ({config.n_head}) must be divisible by "
            f"n_kv_head ({config.n_kv_head})"
        )
        assert config.n_embd % config.n_head == 0, (
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
        )

        self.config = config
        k = config.n_embd // config.n_head  # dimension of each head

        # Q, K, V projection matrices
        self.queries = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # only g heads in GQA
        self.keys = nn.Linear(config.n_embd, config.n_kv_head * k, bias=False)
        # only g heads in GQA
        self.values = nn.Linear(config.n_embd, config.n_kv_head * k, bias=False)

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

        # Q = (B, T, D) -> (B, T, h, k) -> (B, h, T, k)
        Q = self.queries(x).reshape(B, T, h, k).transpose(1, 2)

        # K, V = (B, T, g * k) -> (B, T, g, k) -> (B, g, T, k)
        K = self.keys(x).reshape(B, T, self.config.n_kv_head, k).transpose(1, 2)
        V = self.values(x).reshape(B, T, self.config.n_kv_head, k).transpose(1, 2)

        # Repeat K and V for each head: (B, g, T, k) -> (B, h, T, k)
        K = K.repeat_interleave(h // self.config.n_kv_head, dim=1)
        V = V.repeat_interleave(h // self.config.n_kv_head, dim=1)

        # Softmax along the 4th dim
        A_masked = ((Q @ K.transpose(2, 3)) / (k**0.5)).masked_fill(
            self.att_mask[:, :, :T, :T] == 0, float("-inf")
        )
        A = F.softmax(A_masked, dim=-1)

        # (B, h, T, k) -> (B, T, h, k) -> (B, T, D)
        Y = (A @ V).transpose(1, 2).reshape(B, T, D)
        output = self.out_proj(Y)

        return output
