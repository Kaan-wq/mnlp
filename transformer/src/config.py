from dataclasses import dataclass

@dataclass
class GPTConfig:
    """Configuration for the GPT model."""
    vocab_size: int = 50257
    max_seq_length: int = 512
    n_embd: int = 384
    n_layer: int = 8
    n_head: int = 8
    dropout: float = 0.1
