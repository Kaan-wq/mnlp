from dataclasses import dataclass

@dataclass
class GPTConfig:
    """Configuration for the GPT model."""
    vocab_size:     int   = 50257
    max_seq_length: int   = 256
    n_embd:         int   = 128
    n_layer:        int   = 4
    n_head:         int   = 4
    dropout:        float = 0.1
    device:         str   = "cpu"
