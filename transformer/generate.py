import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer

from src.model import GPT

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPT.from_pretrained("kaanino/gpt-mha-baseline")
model.to(device)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

input_ids = tokenizer(
    "The history of artificial intelligence", return_tensors="pt"
).input_ids.to(device)

output = model.generate(
    input_ids,
    max_new_tokens=50,
    temperature=0.8,
    top_k=50,
    do_sample=True,
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
