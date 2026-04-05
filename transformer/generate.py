import torch
import torch.nn.functional as F
from src.model import GPT
from src.config import GPTConfig
from src.model import GPT

def decode(
        model: GPT,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        greedy: bool = False,
    ) -> torch.Tensor:

    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.max_seq_length else idx[:, -model.config.max_seq_length:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :] / temperature

        # top-k
        top_k_values, _ = torch.topk(logits, k=top_k)
        logits = logits.masked_fill(logits < top_k_values[:, -1].unsqueeze(-1), float('-inf'))

        # top-p
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

        probs = F.softmax(logits, dim=-1)

        if greedy:
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate(model, prompt: str, char_to_idx: dict, idx_to_char: dict, **kwargs) -> str:
    # 1. tokenize
    tokens = [char_to_idx[c] for c in prompt]
    idx = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # (1, seq_len)

    # 2. truncate if needed
    if idx.size(1) > model.config.max_seq_length:
        idx = idx[:, -model.config.max_seq_length:]

    # 3. generate
    output_ids = decode(model, idx, **kwargs)

    # 4. decode back to string, strip the prompt
    output_tokens = output_ids[0, len(tokens):].tolist()
    return ''.join([idx_to_char[i] for i in output_tokens])


def main():
    checkpoint = torch.load("gpt_checkpoint.pth", map_location="cpu")
    
    model  = GPT(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    
    char_to_idx = checkpoint["char_to_idx"]
    idx_to_char = checkpoint["idx_to_char"]

    prompt = "Once upon a time"
    generated_text = generate(
        model, prompt, char_to_idx, idx_to_char,
        max_new_tokens=200, temperature=0.8, top_k=40, top_p=0.9
    )
    print(generated_text)
