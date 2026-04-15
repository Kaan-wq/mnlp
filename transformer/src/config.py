from transformers import PreTrainedConfig


class GPTConfig(PreTrainedConfig):
    model_type = "gpt"

    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_length: int = 128,
        n_embd: int = 64,
        n_layer: int = 4,
        n_head: int = 4,
        n_kv_head: int = 4,
        norm_type: str = "rmsnorm",  # ["layernorm", "rmsnorm"]
        tie_embeddings: bool = True,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.norm_type = norm_type
        self.tie_embeddings = tie_embeddings
        self.dropout = dropout
