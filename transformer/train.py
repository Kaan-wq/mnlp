import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from src.model import GPT
from src.config import GPTConfig
import wandb
import time
from dotenv import load_dotenv
load_dotenv()

@dataclass
class TrainingConfig:
    batch_size:    int   = 32
    max_iters:     int   = 1000
    eval_interval: int   = 100
    eval_iters:    int   = 10
    learning_rate: float = 3e-4
    data_path:     str   = "data/input.txt"
    device:        str   = "cpu"


def load_data(path: str, device: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    text  = open(path).read()
    chars = sorted(set(text))
    stoi  = {ch: i for i, ch in enumerate(chars)}
    data  = torch.as_tensor([stoi[ch] for ch in text], dtype=torch.long, device=device)
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
    wandb.init(
        project="gpt-from-scratch",
        tags=["test","debugging", "first-run"],
        config={**config.__dict__, **t_config.__dict__},
    )

    train_data, val_data, vocab_size = load_data(t_config.data_path, config.device)
    config.vocab_size = vocab_size

    model     = GPT(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t_config.learning_rate)

    for step in range(t_config.max_iters):
        with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
            if step % t_config.eval_interval == 0:
                val_loss = evaluate(model, val_data, config, t_config, config.device)
                wandb.log({"val_loss": val_loss, "step": step})

            x, y   = get_batch(train_data, config, t_config, config.device)
            t0 = time.time()
            logits = model.forward(x)
            loss   = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        t1 = time.time()
        tokens_per_second = (t_config.batch_size * config.max_seq_length) / (t1 - t0 + 1e-8)

        wandb.log({
            "train_loss": loss.item(),
            "grad_norm": grad_norm,
            "step_duration_ms": (t1 - t0) * 1000,
            "tokens_per_second": tokens_per_second,
            "step": step
        })


    torch.save(model.state_dict(), "gpt_model.pth")
    print("Training complete.")


if __name__ == "__main__":
    config   = GPTConfig()
    t_config = TrainingConfig()
    train(config, t_config)
