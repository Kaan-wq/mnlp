import torch
from peft import LoraConfig, get_peft_model
from transformers import MarianMTModel, MarianTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, concatenate_datasets, DatasetDict
import evaluate
from dotenv import load_dotenv
import numpy as np
load_dotenv()


def main():
    raw_datasets = load_dataset(
        "Helsinki-NLP/opus-100", "en-fr")

    merged_datasets = concatenate_datasets(
        [raw_datasets['train'], raw_datasets['validation']])
    merged_datasets = concatenate_datasets(
        [merged_datasets, raw_datasets['test']])

    split_train_testeval = merged_datasets.train_test_split(
        test_size=0.2, seed=20)
    split_eval_test = split_train_testeval['test'].train_test_split(
        test_size=0.5, seed=20)
    split_datasets = DatasetDict({
        'train': split_train_testeval['train'],
        'validation': split_eval_test['train'],
        'test': split_eval_test['test']
    })

    checkpoint = "Helsinki-NLP/opus-mt-en-fr"
    tokenizer = MarianTokenizer.from_pretrained(checkpoint)
    model = MarianMTModel.from_pretrained(checkpoint)
    model.enable_input_require_grads()
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_preds = tokenizer.batch_decode(
            predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds,
                                references=decoded_labels)
        return {"bleu": result["score"]}

    def preprocess_function(examples):
        inputs = [ex["en"] for ex in examples['translation']]
        targets = [ex["fr"] for ex in examples['translation']]

        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=128, truncation=True)
        return model_inputs

    tokenized_datasets = split_datasets.map(
        preprocess_function, batched=True, remove_columns=split_datasets["train"].column_names)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="SEQ_2_SEQ_LM",
    )
    lora_model = get_peft_model(model, peft_config)
    lora_model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
        output_dir="test-trainer",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs=3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=4,
        fp16=torch.cuda.is_available(),
        learning_rate=3e-4,
        weight_decay=0.01,
        gradient_checkpointing=True,
        predict_with_generate=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        dataloader_num_workers=4,
        push_to_hub=True,
        report_to="wandb",
        run_name="marianmt-en-fr_lora",
    )

    trainer = Seq2SeqTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )
    # results = trainer.evaluate(max_length=128)
    # print(results)
    trainer.train()


if __name__ == "__main__":
    main()
