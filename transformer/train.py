import torch
import torch.nn.functional as F
from dataclasses import dataclass
from src.model import GPT
from src.config import GPTConfig


@dataclass
class TrainingConfig:
    batch_size:    int   = 64
    max_iters:     int   = 5000
    eval_interval: int   = 500
    eval_iters:    int   = 200
    learning_rate: float = 3e-4
    data_path:     str   = "data/input.txt"


def load_data(path: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    text  = open(path).read()
    chars = sorted(set(text))
    stoi  = {ch: i for i, ch in enumerate(chars)}
    data  = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    n     = int(0.9 * len(data))
    return data[:n], data[n:], len(chars)


def get_batch(
        data: torch.Tensor,
        config: GPTConfig,
        t_config: TrainingConfig,
        device
    ) -> tuple[torch.Tensor, torch.Tensor]:

    ix = torch.randint(len(data) - config.max_seq_length, (t_config.batch_size,))
    x  = torch.stack([data[i   : i+config.max_seq_length  ] for i in ix]).to(device)
    y  = torch.stack([data[i+1 : i+config.max_seq_length+1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def evaluate(
        model: GPT,
        val_data: torch.Tensor,
        config: GPTConfig,
        t_config: TrainingConfig,
        device
    ) -> float:

    model.eval()
    total_loss = 0.0
    for _ in range(t_config.eval_iters):
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            x, y       = get_batch(val_data, config, t_config, device)
            logits     = model.forward(x)
            loss       = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    model.train()
    return total_loss / t_config.eval_iters


def train(config: GPTConfig, t_config: TrainingConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, val_data, vocab_size = load_data(t_config.data_path)
    config.vocab_size = vocab_size

    model     = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t_config.learning_rate)

    for step in range(t_config.max_iters):
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            if step % t_config.eval_interval == 0:
                val_loss = evaluate(model, val_data, config, t_config, device)
                print(f"step {step} | val loss {val_loss:.4f}")

            x, y   = get_batch(train_data, config, t_config, device)
            logits = model.forward(x)
            loss   = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "gpt_model.pth")
    print("Training complete.")


if __name__ == "__main__":
    config   = GPTConfig()
    t_config = TrainingConfig()
    train(config, t_config)
