import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .blocks import TransformerBlock
from .config import GPTConfig


class GPT(PreTrainedModel, GenerationMixin):
    config_class = GPTConfig

    def __init__(self, config: GPTConfig) -> None:
        super().__init__(config)

        if config.pos_enc_type == "absolute":
            self.pos_embd = nn.Embedding(config.max_seq_length, config.n_embd)
        else:
            self.pos_embd = None
        self.token_embd = nn.Embedding(config.vocab_size, config.n_embd)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.logits_proj = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights to avoid exploding loss at the beginning of training
        self.apply(self._init_weights)

        #  Tie weights
        self.post_init()

    def get_input_embeddings(self):
        return self.token_embd

    def set_input_embeddings(self, value):
        self.token_embd = value

    def get_output_embeddings(self):
        return self.logits_proj

    def set_output_embeddings(self, value):
        self.logits_proj = value

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "_is_residual_proj"):
                std = 0.02 / math.sqrt(2 * self.config.n_layer)
            nn.init.trunc_normal_(
                module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            std = 0.02
            nn.init.trunc_normal_(
                module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std
            )
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        num_items_in_batch: int | None = None,
        past_key_values: tuple | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]  # only last token

        if self.pos_embd is not None:
            offset = (
                past_key_values[0][0].shape[2] if past_key_values is not None else 0
            )
            pos = torch.arange(
                offset, offset + input_ids.size(1), device=input_ids.device
            ).unsqueeze(0)
            x = self.token_embd(input_ids) + self.pos_embd(pos)
        else:
            x = self.token_embd(input_ids)

        new_past_key_values = []
        for i, block in enumerate(self.transformer_blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, new_kv = block(x, past_kv=past_kv, use_cache=use_cache)
            new_past_key_values.append(new_kv)

        logits = self.logits_proj(self.ln_f(x))

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum" if num_items_in_batch is not None else "mean",
            )
            if num_items_in_batch is not None:
                loss = loss / num_items_in_batch

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=tuple(new_past_key_values) if use_cache else None,
        )
