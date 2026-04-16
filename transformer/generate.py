import torch
from transformers import AutoTokenizer

from src.gpt_model import GPT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATHS = [
    "kaanino/gpt-mha-RoPE",
    "kaanino/gpt-mqa-RoPE",
    "kaanino/gpt-gqa-RoPE",
]
PROMPT = "The history of artificial intelligence"


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
    generated = output_ids[0][input_ids.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def _run_timed(model, input_ids, max_new_tokens, use_cache, n_runs):
    """Run generation n_runs times and return avg tokens/sec."""
    # warmup
    with torch.no_grad():
        model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            do_sample=False,
            pad_token_id=50256,
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
                use_cache=use_cache,
                do_sample=False,
                pad_token_id=50256,
            )
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000)
    return max_new_tokens / (sum(times) / n_runs)


def benchmark_cache(model, max_new_tokens=20, prompt_lengths=None, n_runs=3):
    """Compare no-cache vs cache across prompt lengths."""
    if prompt_lengths is None:
        prompt_lengths = [8, 16, 32, 64]

    print(
        f"\n  {'Prompt len':<14} {'No cache (tok/s)':<20} "
        f"{'Cache (tok/s)':<18} {'Speedup':>8}"
    )
    print(f"  {'-' * 62}")

    for prompt_len in prompt_lengths:
        # stay within max_seq_length
        if prompt_len + max_new_tokens > model.config.max_seq_length:
            continue
        input_ids = torch.randint(0, model.config.vocab_size, (1, prompt_len)).to(
            DEVICE
        )
        tps_no_cache = _run_timed(
            model, input_ids, max_new_tokens, use_cache=False, n_runs=n_runs
        )
        tps_cache = _run_timed(
            model, input_ids, max_new_tokens, use_cache=True, n_runs=n_runs
        )
        speedup = tps_cache / tps_no_cache
        print(
            f"  {prompt_len:<14} {tps_no_cache:<20.1f} "
            f"{tps_cache:<18.1f} {speedup:>7.2f}x"
        )


if __name__ == "__main__":
    print("=" * 65)
    print("  KV Cache Benchmark")
    print("=" * 65)

    # Generation sanity check
    model, tokenizer = load_model(MODEL_PATHS[1])
    print("\n=== Generation sample ===")
    output = generate(model, tokenizer, PROMPT, max_new_tokens=50, temperature=0.8)
    print(f"  Prompt : {PROMPT}")
    print(f"  Output : {output}")

    # Cache speedup vs prompt length (single model)
    print("\n=== Cache speedup vs prompt length (MHA) ===")
    benchmark_cache(
        model, max_new_tokens=20, prompt_lengths=[8, 16, 32, 64], n_runs=100
    )
    del model
    torch.cuda.empty_cache()
