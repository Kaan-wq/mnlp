from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch


def preprocess_data(tokenizer, dataset):
    def preprocess_function(examples):
        return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, return_tensors='pt')
    return dataset.map(preprocess_function, batched=True)


def main():
    checkpoint = "bert-base-uncased"
    raw_datasets = load_dataset('glue', 'mrpc')

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    tokenized_datasets = preprocess_data(tokenizer, raw_datasets).remove_columns(['sentence1', 'sentence2', 'idx'])
    data_collator = DataCollatorWithPadding(tokenizer, return_tensors='pt')

    print(tokenized_datasets)

    #train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=8, collate_fn=data_collator)
    #eval_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=8, collate_fn=data_collator)
    #test_dataloader = DataLoader(tokenized_datasets['test'], batch_size=8, collate_fn=data_collator)



if __name__ == "__main__":
    main()
