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


def main():
    set_seed(SEED)

    RUN_NAME = "gpt-mqa-baseline"
    BATCH_SIZE, GRAD_ACC_STEPS, MAX_SEQ_LEN = 64, 8, 256
    TOKENS_PER_STEP = BATCH_SIZE * GRAD_ACC_STEPS * MAX_SEQ_LEN
    DATASET_TOKENS = 103_000_000
    STEPS_DATASET = DATASET_TOKENS // TOKENS_PER_STEP
    print(f"Dataset steps: {STEPS_DATASET:,}")

    # Create tokenizer and data collator
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Create model
    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_length=MAX_SEQ_LEN,
        n_embd=128,
        n_layer=4,
        n_head=2,
        attn_type="mqa",
        n_kv_head=2,
        dropout=0.1,
    )
    model = GPT(model_config)
    print(f"Expected Initial Loss: {math.log(model_config.vocab_size):.3f}")

    non_embd_params = model.num_parameters(exclude_embeddings=True)
    print(f"Non-embedding parameters: {non_embd_params:,}")
    print(f"Tokens needed (Chinchilla): {non_embd_params * 20:,}")

    STEPS_OPT = non_embd_params * 20 // TOKENS_PER_STEP
    print(f"Estimated optimal number of steps: {STEPS_OPT:,}")

    # Model with X parameters should be trained on Y ≃ 20 * X tokens
    # https://arxiv.org/abs/2203.15556

    # We will be comparing architecture so we must set the same budget for all models.
    STEPS = STEPS_DATASET

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
        eval_strategy="steps",
        eval_steps=int(STEPS * 0.1),  # evaluate every 10% of steps
        save_strategy="steps",
        save_steps=int(STEPS * 0.1),  # save every 10% of steps
        load_best_model_at_end=True,
        num_train_epochs=1,  # overriden by max_steps
        max_steps=STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        fp16=torch.cuda.is_available(),
        bf16=False,
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
        callbacks=[PerplexityCallback()],
    )
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
