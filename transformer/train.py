import random
import numpy as np
import torch
import math
from transformers import Trainer, DataCollatorForLanguageModeling, TrainingArguments, AutoTokenizer
from src.model import GPT
from src.config import GPTConfig
from datasets import load_dataset, concatenate_datasets, DatasetDict
from dotenv import load_dotenv
load_dotenv()

SEED = 20


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)

    # Create model
    model_config = GPTConfig()
    model = GPT(model_config)
    print(f"Parameters: {model.num_parameters():,}")

    # Create tokenizer and data collator
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=model_config.max_seq_length
        )

    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenized_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names)

    RUN_NAME = "gpt-mha-baseline"

    training_args = TrainingArguments(
        output_dir=RUN_NAME,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs=3,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=4,
        fp16=torch.cuda.is_available(),
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_checkpointing=True,
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
