import evaluate
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

load_dotenv()


def preprocess_data(tokenizer, dataset):
    def preprocess_function(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

    return dataset.map(preprocess_function, batched=True)


def main():
    checkpoint = "bert-base-uncased"
    raw_datasets = load_dataset("glue", "mrpc")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    tokenized_datasets = preprocess_data(tokenizer, raw_datasets).remove_columns(
        ["sentence1", "sentence2", "idx"]
    )
    data_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")

    metric = evaluate.load("glue", "mrpc")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="test-trainer",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        fp16=torch.cuda.is_available(),
        learning_rate=2e-5,
        weight_decay=0.01,
        report_to="wandb",
        run_name="bert-mrpc-fine-tuning",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()


if __name__ == "__main__":
    main()
