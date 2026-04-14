import math
import random

import numpy as np
import torch
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.config import GPTConfig
from src.model import GPT

load_dotenv()

SEED = 20


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if "eval_loss" in metrics:
            wandb.log(
                {
                    "eval/perplexity": math.exp(metrics["eval_loss"]),
                    "step": state.global_step,
                }
            )


class Step1DebugCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            self._debug = True

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 1:
            print("\n=== STEP 1 DEBUG ===")
            print(f"Model dtype:        {next(model.parameters()).dtype}")
            print(f"Model device:       {next(model.parameters()).device}")
            for name, p in model.named_parameters():
                if p.grad is not None:
                    print(
                        f"Grad {name[:40]:40s} norm={p.grad.norm():.4f} "
                        f"has_nan={p.grad.isnan().any().item()}"
                    )
                    break  # just the first one


class Step1LossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step <= 3:
            print(f"\nStep {state.global_step} raw logs: {logs}")


def main():
    set_seed(SEED)

    RUN_NAME = "gpt-mha-baseline"
    BATCH_SIZE, GRAD_ACC_STEPS, MAX_SEQ_LEN = 64, 8, 256
    TOKENS_PER_STEP = BATCH_SIZE * GRAD_ACC_STEPS * MAX_SEQ_LEN
    DATASET_TOKENS = 103_000_000
    DATASET_STEPS = DATASET_TOKENS // TOKENS_PER_STEP
    print(f"Dataset steps: {DATASET_STEPS:,}")

    # Create tokenizer and data collator
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    print(tokenizer.vocab_size)

    # Create model
    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_length=MAX_SEQ_LEN,
        n_embd=128,
        n_layer=6,
        n_head=2,
        attn_type="mha",
    )
    model = GPT(model_config)
    print(f"Expected Initial Loss: {math.log(model_config.vocab_size):.3f}")

    non_embd_params = model.num_parameters(exclude_embeddings=True)
    print(f"Non-embedding parameters: {non_embd_params:,}")
    print(f"Tokens needed (Chinchilla): {non_embd_params * 20:,}")

    STEPS = non_embd_params * 20 // TOKENS_PER_STEP
    print(f"Estimated steps: {STEPS:,}")

    # Model with X parameters should be trained on Y ≃ 20 * X tokens
    # https://arxiv.org/abs/2203.15556

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=False, add_special_tokens=True)

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples}
        total_length = len(concatenated[list(examples.keys())[0]])
        if total_length >= model_config.max_seq_length:
            total_length = (
                total_length // model_config.max_seq_length
            ) * model_config.max_seq_length
        return {
            k: [
                t[i : i + model_config.max_seq_length]
                for i in range(0, total_length, model_config.max_seq_length)
            ]
            for k, t in concatenated.items()
        }

    raw_datasets = load_dataset("wikitext", "wikitext-103-raw-v1")
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=raw_datasets["train"].column_names,
    )
    tokenized_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=4)

    training_args = TrainingArguments(
        output_dir=RUN_NAME,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs=1,  # overriden by max_steps
        max_steps=STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        fp16=False,
        # bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_steps=int(STEPS * 0.1),  # 10% of steps for warmup
        weight_decay=0.01,
        gradient_checkpointing=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=2,
        push_to_hub=True,
        report_to="wandb",
        run_name=RUN_NAME,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=None,
        callbacks=[PerplexityCallback(), Step1DebugCallback(), Step1LossCallback()],
    )

    # ── Real batch sanity check ───────────────────────────────────────────
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    model.eval()
    model = model.float()  # force float32 regardless of training precision

    sample_loader = DataLoader(
        tokenized_datasets["train"],
        batch_size=4,
        collate_fn=data_collator,
    )
    batch = next(iter(sample_loader))

    input_ids = batch["input_ids"].to(model.device)
    labels = batch["labels"].to(model.device)

    print(f"input_ids shape:        {input_ids.shape}")
    print(f"labels shape:           {labels.shape}")
    print(
        f"input_ids min/max:      {input_ids.min().item()} / {input_ids.max().item()}"
    )
    print(f"labels unique values:   {labels.unique().numel()} unique tokens")
    print(f"labels has -100:        {(labels == -100).any().item()}")
    print(f"fraction -100:          {(labels == -100).float().mean().item():.3f}")

    with torch.no_grad():
        out = model(input_ids, labels=labels)
        print(f"\nLoss on real batch:     {out.loss.item():.3f}")
        print(f"Expected (uniform):     {math.log(model_config.vocab_size):.3f}")

        shift_logits = out.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute loss manually token by token to check for outliers
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        )
        print(f"\nPer-token loss mean:    {per_token_loss.mean().item():.3f}")
        print(f"Per-token loss max:     {per_token_loss.max().item():.3f}")
        print(f"Per-token loss min:     {per_token_loss.min().item():.3f}")
        print(f"Any inf in loss:        {per_token_loss.isinf().any().item()}")
        print(f"Any nan in loss:        {per_token_loss.isnan().any().item()}")
    # ─────────────────────────────────────────────────────────────────────

    print(f"Model dtype before train(): {next(model.parameters()).dtype}")
    print(f"GPUs available: {torch.cuda.device_count()}")
    trainer.train()

    trainer.create_model_card(
        language="en",
        license="apache-2.0",
        tags=["gpt", "causal-lm", "from-scratch", "wikitext-2"],
        model_name=RUN_NAME,
        tasks="text-generation",
        dataset_tags="wikitext",
        dataset="wikitext-2-raw-v1",
    )
    trainer.push_to_hub()


if __name__ == "__main__":
    main()
