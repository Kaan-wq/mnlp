import torch
import torch.nn as nn
from config import GPTConfig
from blocks import TransformerBlock

class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.pos_embd = nn.Embedding(config.max_seq_length, config.n_embd)
        self.token_embd = nn.Embedding(config.vocab_size, config.n_embd)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.logits_proj = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.token_embd(x) + self.pos_embd(pos)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.logits_proj(x)
