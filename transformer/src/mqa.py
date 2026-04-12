import torch.nn as nn
import torch.nn.functional as F
import torch
from .config import GPTConfig


class MaskedMultiQuerySelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        self.config = config
        k = config.n_embd // config.n_head  # dimension of each head

        # Q, K, V projection matrices
        self.queries = nn.Linear(
            config.n_embd, config.n_embd, bias=False, device=config.device)
        self.keys = nn.Linear(config.n_embd, k,
                              bias=False, device=config.device)  # only one head in MQA
        self.values = nn.Linear(
            config.n_embd, k, bias=False, device=config.device)  # only one head in MQA

        self.out_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=False, device=config.device)

        self.register_buffer(
            "att_mask",
            torch.tril(
                torch.ones(config.max_seq_length,
                           config.max_seq_length, device=config.device)
            ).reshape(1, 1, config.max_seq_length, config.max_seq_length)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.size()
        h = self.config.n_head
        k = D // h

        # Q = (B, T, D) -> (B, T, h, k) -> (B, h, T, k)
        Q = self.queries(x).reshape(B, T, h, k).transpose(1, 2)

        # K, V = (B, T, k) -> (B, T, 1, k) -> (B, 1, T, k)
        K = self.keys(x).reshape(B, T, 1, k).transpose(1, 2)
        V = self.values(x).reshape(B, T, 1, k).transpose(1, 2)

        # Softmax along the 4th dim
        A_masked = ((Q @ K.transpose(2, 3)) / (k ** 0.5)
                    ).masked_fill(self.att_mask[:, :, :T, :T] == 0, float("-inf"))
        A = F.softmax(A_masked, dim=-1)

        # (B, h, T, k) -> (B, T, h, k) -> (B, T, D)
        Y = (A @ V).transpose(1, 2).reshape(B, T, D)
        output = self.out_proj(Y)

        return output
