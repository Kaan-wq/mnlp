"""
Minimal reproduction of HuggingFace Trainer loss logging bug
with gradient_accumulation_steps > 1.

Expected: logged loss ≈ ln(vocab_size) at step 1
Actual:   logged loss ≈ ln(vocab_size) × gradient_accumulation_steps

Tested with:
- transformers==4.x.x  (fill in your version)
- torch==2.x.x
- Python 3.12
"""

import math

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

GRAD_ACC_STEPS = 8

# sshleifer/tiny-gpt2: 2 layers, 2 heads, hidden=64 — smallest GPT-2 variant on the hub
MODEL_ID = "sshleifer/tiny-gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

VOCAB_SIZE = tokenizer.vocab_size
print(f"Vocab size:        {VOCAB_SIZE}")
print(f"Expected loss:     {math.log(VOCAB_SIZE):.4f}  (ln({VOCAB_SIZE}))\n")

# --- Minimal dataset: random token sequences ---
SEQ_LEN = 32
N_SAMPLES = 256
data = {"input_ids": torch.randint(0, VOCAB_SIZE, (N_SAMPLES, SEQ_LEN)).tolist()}
dataset = Dataset.from_dict(data)
dataset = dataset.map(lambda x: {"labels": x["input_ids"]})


# --- Capture logged loss ---
class LogCaptureCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            expected = math.log(VOCAB_SIZE)
            reported = logs["loss"]
            ratio = reported / expected
            print(f"[step {state.global_step}]")
            print(f"  Reported loss:  {reported:.4f}")
            print(f"  Expected loss:  {expected:.4f}")
            print(
                f"  Ratio:          {ratio:.2f}  ← should be 1.0, "
                f"actual is ~{GRAD_ACC_STEPS} (= gradient_accumulation_steps)\n"
            )


training_args = TrainingArguments(
    output_dir="repro-grad-acc-bug",
    max_steps=20,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=GRAD_ACC_STEPS,
    logging_steps=1,
    report_to="none",
    fp16=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    callbacks=[LogCaptureCallback()],
)
trainer.train()
