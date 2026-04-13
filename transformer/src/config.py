from transformers import PreTrainedConfig


class GPTConfig(PreTrainedConfig):
    model_type = "gpt"

    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_length: int = 64,
        n_embd: int = 64,
        n_layer: int = 4,
        n_head: int = 2,
        attn_type: str = "mha",
        n_kv_head: int = 2,
        dropout: float = 0.1,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.attn_type = attn_type
        self.n_kv_head = n_kv_head
        self.dropout = dropout
