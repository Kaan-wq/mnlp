from dataclasses import dataclass

@dataclass
class GPTConfig:
    """Configuration for the GPT model."""
    vocab_size: int = 50257
    max_seq_length: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    dropout: float = 0.1
