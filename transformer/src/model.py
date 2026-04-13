import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutput
from .config import GPTConfig
from .blocks import TransformerBlock
from transformers import PreTrainedModel
import torch.nn.functional as F
from .types import AttentionType


class GPT(PreTrainedModel):
    config_class = GPTConfig

    def __init__(self, config: GPTConfig) -> None:
        super().__init__(config)

        attn_type = AttentionType(config.attn_type)
        self.pos_embd = nn.Embedding(config.max_seq_length, config.n_embd)
        self.token_embd = nn.Embedding(config.vocab_size, config.n_embd)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config, attn_type) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.logits_proj = nn.Linear(
            config.n_embd, config.vocab_size, bias=False)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs
    ) -> CausalLMOutput:
        pos = torch.arange(input_ids.size(
            1), device=input_ids.device).unsqueeze(0)
        x = self.token_embd(input_ids) + self.pos_embd(pos)

        for block in self.transformer_blocks:
            x = block(x)

        logits = self.logits_proj(self.ln_f(x))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1))

        return CausalLMOutput(loss=loss, logits=logits)
