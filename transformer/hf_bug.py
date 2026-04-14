"""
Minimal reproduction of HuggingFace Trainer loss logging bug
with gradient_accumulation_steps > 1.

Expected: logged loss ≈ ln(vocab_size) ≈ 10.82 at step 1
Actual:   logged loss ≈ ln(vocab_size) × gradient_accumulation_steps

Tested with:
- transformers==4.x.x  (fill in your version)
- torch==2.x.x
- Python 3.12
"""

import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.modeling_outputs import CausalLMOutput

VOCAB_SIZE = 1000
SEQ_LEN = 32
GRAD_ACC_STEPS = 16


# --- Minimal model ---
class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, 64)
        self.head = nn.Linear(64, VOCAB_SIZE, bias=False)

    def forward(self, input_ids, labels=None, **kwargs):
        logits = self.head(self.emb(input_ids))
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, VOCAB_SIZE),
                labels.view(-1),
            )

        return CausalLMOutput(loss=loss, logits=logits)


# --- Minimal dataset ---
class RandomTokenDataset(Dataset):
    def __init__(self, size=256):
        self.data = torch.randint(0, VOCAB_SIZE, (size, SEQ_LEN))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"input_ids": self.data[idx], "labels": self.data[idx]}


# --- Capture logged loss ---
class LogCaptureCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            expected = math.log(VOCAB_SIZE)
            reported = logs["loss"]
            ratio = reported / expected
            print(f"\n[step {state.global_step}]")
            print(f"  Reported loss:  {reported:.4f}")
            print(f"  Expected loss:  {expected:.4f}  (ln({VOCAB_SIZE}))")
            print(
                f"  Ratio:          {ratio:.2f}  ← should be 1.0, "
                f"actual is ~{GRAD_ACC_STEPS} (= gradient_accumulation_steps)"
            )


model = TinyLM()
# Verify model is near uniform at init
with torch.no_grad():
    dummy = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))
    out = model(dummy, labels=dummy)
    print(f"True initial loss: {out.loss.item():.4f}")
    print(f"Expected:          {math.log(VOCAB_SIZE):.4f}\n")

training_args = TrainingArguments(
    output_dir="repro-grad-acc-bug",
    max_steps=20,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=GRAD_ACC_STEPS,  # bug only appears when > 1
    logging_steps=1,
    report_to="none",
    fp16=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=RandomTokenDataset(),
    callbacks=[LogCaptureCallback()],
)
trainer.train()
