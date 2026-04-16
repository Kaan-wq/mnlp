import torch
from transformers import AutoTokenizer

from src.gpt_model import GPT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path: str) -> tuple[GPT, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT.from_pretrained(model_path)
    model.eval()
    model.to(DEVICE)
    return model, tokenizer


def generate(
    model: GPT,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    do_sample: bool = True,
) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    generated = output_ids[0][input_ids.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def benchmark(
    model: GPT,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    n_runs: int = 5,
) -> float:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    # Warmup
    with torch.no_grad():
        model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []

    for _ in range(n_runs):
        start.record()
        with torch.no_grad():
            model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                use_cache=False,
            )
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000)  # ms → seconds

    avg_time = sum(times) / len(times)
    tokens_per_sec = max_new_tokens / avg_time
    print(f"  Avg time       {avg_time:.3f}s")
    print(f"  Tokens/sec     {tokens_per_sec:.1f}")
    return tokens_per_sec


if __name__ == "__main__":
    MODEL_PATH = "kaanino/gpt-mha-RoPE"
    PROMPT = "The history of artificial intelligence"

    model, tokenizer = load_model(MODEL_PATH)
    print(type(model))
    print(isinstance(model, GPT))

    print("\n=== Generation ===")
    output = generate(model, tokenizer, PROMPT, max_new_tokens=100, temperature=0.8)
    print(f"Prompt : {PROMPT}")
    print(f"Output : {output}")

    print("\n=== Benchmark ===")
    benchmark(model, tokenizer, PROMPT, max_new_tokens=100, n_runs=5)
