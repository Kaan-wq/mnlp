from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Configuration for the GPT model."""
    vocab_size:     int = 50257
    max_seq_length: int = 64
    n_embd:         int = 64
    n_layer:        int = 4
    n_head:         int = 2
    n_kv_head:      int = 2
    dropout:        float = 0.1
    device:         str = "cpu"
