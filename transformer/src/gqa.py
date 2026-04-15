import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


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
        self.k = config.n_embd // config.n_head  # dimension of each head

        # Q, K, V projection matrices
        self.queries = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # only g heads in GQA
        self.keys = nn.Linear(config.n_embd, config.n_kv_head * self.k, bias=False)
        # only g heads in GQA
        self.values = nn.Linear(config.n_embd, config.n_kv_head * self.k, bias=False)

        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj._is_residual_proj = True  # mark for scaled init

        self.register_buffer(
            "att_mask",
            torch.tril(
                torch.ones(config.max_seq_length, config.max_seq_length)
            ).reshape(1, 1, config.max_seq_length, config.max_seq_length),
        )

        if config.pos_enc_type == "relative":
            angles = torch.pow(10000, -2 * torch.arange(0, self.k // 2) / self.k)
            positions = torch.arange(config.max_seq_length).unsqueeze(1)
            angle_matrix = positions * angles  # (T, k//2)
            # repeat to cover full head dim — same frequencies apply to both halves
            self.register_buffer(
                "cos_embd", torch.cos(angle_matrix).repeat(1, 2)
            )  # (T, k)
            self.register_buffer(
                "sin_embd", torch.sin(angle_matrix).repeat(1, 2)
            )  # (T, k)

    def _apply_rope(self, x: torch.Tensor, T: int) -> torch.Tensor:
        cos = self.cos_embd[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, k)
        sin = self.sin_embd[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, k)
        return x * cos + rotate_half(x) * sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.size()
        h, k, g = self.config.n_head, self.k, self.config.n_kv_head

        # Q = (B, T, D) -> (B, T, h, k) -> (B, h, T, k)
        Q = self.queries(x).reshape(B, T, h, k).transpose(1, 2)

        # K, V = (B, T, g * k) -> (B, T, g, k) -> (B, g, T, k)
        K = self.keys(x).reshape(B, T, g, k).transpose(1, 2)
        V = self.values(x).reshape(B, T, g, k).transpose(1, 2)

        # Repeat K and V for each head: (B, g, T, k) -> (B, h, T, k)
        K = K.repeat_interleave(h // g, dim=1)
        V = V.repeat_interleave(h // g, dim=1)

        if self.config.pos_enc_type == "relative":
            Q = self._apply_rope(Q, T)
            K = self._apply_rope(K, T)

        # Softmax along the 4th dim
        A_masked = ((Q @ K.transpose(2, 3)) / (k**0.5)).masked_fill(
            self.att_mask[:, :, :T, :T] == 0, float("-inf")
        )
        A = F.softmax(A_masked, dim=-1)

        # (B, h, T, k) -> (B, T, h, k) -> (B, T, D)
        Y = (A @ V).transpose(1, 2).reshape(B, T, D)

        return self.out_proj(Y)
