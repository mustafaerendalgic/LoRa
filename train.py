
#DIVERSE TRAINING

import os
import gc
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, DataCollatorForLanguageModeling,
    Trainer, TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType

torch.cuda.empty_cache()
gc.collect()

print("🚀 Final DIVERSE Training (Train-Loss Early Stop) Başlıyor...")

# ==========================================
# 1. BASE MODEL & TOKENIZER
# ==========================================
model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

# ==========================================
# 2. DATASET
# ==========================================
dataset_path = "/content/CodeGen-Diverse-5K.jsonl"
raw = load_dataset("json", data_files=dataset_path, split="train")

train_ds = raw.filter(lambda x: x["split"] == "train")
val_ds   = raw.filter(lambda x: x["split"] == "valid")
test_ds  = raw.filter(lambda x: x["split"] == "test")

print(f"Train: {len(train_ds)} | Validation: {len(val_ds)} | Test: {len(test_ds)}")

# ==========================================
# 3. PREPROCESS
# ==========================================
def preprocess(example):
    system_prompt = (
        "You are an expert Python programmer. "
        "Please read the problem carefully before writing any Python code.\n"
    )
    example["text"] = system_prompt + example["solution"] + tokenizer.eos_token
    return example

train_ds = train_ds.map(preprocess)
val_ds   = val_ds.map(preprocess)
test_ds  = test_ds.map(preprocess)

# ==========================================
# 4. TOKENIZATION
# ==========================================
def tokenize(example):
    out = tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,
        padding="max_length"
    )
    out["labels"] = out["input_ids"].copy()
    return out

train_tok = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
val_tok   = val_ds.map(tokenize, batched=True, remove_columns=val_ds.column_names)
test_tok  = test_ds.map(tokenize, batched=True, remove_columns=test_ds.column_names)

# ==========================================
# 5. LoRA CONFIG
# ==========================================
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==========================================
# 6. TRAINING ARGUMENTS
# ==========================================
training_args = TrainingArguments(
    output_dir="./diverse_lora_checkpoints",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=20,
    save_steps=20,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    num_train_epochs=3,
    weight_decay=0.1,
    fp16=True,
    gradient_checkpointing=True,
    report_to="none",
    optim="adamw_torch_fused"
)

# ==========================================
# 7. TEST LOSS (LOGGING ONLY)
# ==========================================
def compute_test_loss(model, dataloader):
    model.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            steps += 1

    return total_loss / steps

test_dataloader = torch.utils.data.DataLoader(
    test_tok,
    batch_size=1,
    collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# ==========================================
# 8. CALLBACK (TRAIN-LOSS EARLY STOP + CKPT RENAME)
# ==========================================
class TestEarlyStopCallback(TrainerCallback):
    def __init__(self, test_dataloader, patience=3, log_file="train_val_test_log.json"):
        self.test_dataloader = test_dataloader
        self.patience = patience
        self.best_test_loss = float("inf")
        self.counter = 0
        self.log_file = log_file

        self.train_loss_by_step = {}
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.train_loss_by_step[state.global_step] = logs["loss"]

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        step = state.global_step

        # compute TEST loss
        test_loss = compute_test_loss(model, self.test_dataloader)
        train_loss = self.train_loss_by_step.get(step)
        val_loss = metrics.get("eval_loss")

        entry = {
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss
        }
        self.logs.append(entry)

        with open(self.log_file, "w") as f:
            json.dump(self.logs, f, indent=2)

        # ===============================
        # EARLY STOPPING — TEST LOSS ONLY
        # ===============================
        if test_loss < self.best_test_loss:
            self.best_test_loss = test_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("🛑 Early stopping (TEST loss)")
                control.should_training_stop = True

        return control
# ==========================================
# 9. TRAINER
# ==========================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[
        TestEarlyStopCallback(
            test_dataloader=test_dataloader,
            patience=3,
            log_file="diverse_train_val_test_log.json"
        )
    ]
)

print("🚀 Eğitim başlıyor...")
trainer.train()

trainer.save_model("./final_diverse_model_train_based")

print("✅ Eğitim tamamlandı")
print("📄 Log dosyası: diverse_train_val_test_log.json")

#-----------------------------------------------------------

DEEP TRAINING

import os
import gc
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, DataCollatorForLanguageModeling,
    Trainer, TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType

torch.cuda.empty_cache()
gc.collect()

print("🚀 Final DEEP Training (Train-Loss Early Stop) Başlıyor...")

# ==========================================
# 1. BASE MODEL & TOKENIZER
# ==========================================
model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

# ==========================================
# 2. DATASET (DEEP)
# ==========================================
dataset_path = "/content/CodeGen-Deep-5K.jsonl"
raw = load_dataset("json", data_files=dataset_path, split="train")

train_ds = raw.filter(lambda x: x["split"] == "train")
val_ds   = raw.filter(lambda x: x["split"] == "valid")
test_ds  = raw.filter(lambda x: x["split"] == "test")

print(f"Train: {len(train_ds)} | Validation: {len(val_ds)} | Test: {len(test_ds)}")

# ==========================================
# 3. PREPROCESS (solution only)
# ==========================================
def preprocess(example):
    system_prompt = (
        "You are an expert Python programmer. "
        "Please read the problem carefully before writing any Python code.\n"
    )
    example["text"] = system_prompt + example["solution"] + tokenizer.eos_token
    return example

train_ds = train_ds.map(preprocess)
val_ds   = val_ds.map(preprocess)
test_ds  = test_ds.map(preprocess)

# ==========================================
# 4. TOKENIZATION
# ==========================================
def tokenize(example):
    out = tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,
        padding="max_length"
    )
    out["labels"] = out["input_ids"].copy()
    return out

train_tok = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
val_tok   = val_ds.map(tokenize, batched=True, remove_columns=val_ds.column_names)
test_tok  = test_ds.map(tokenize, batched=True, remove_columns=test_ds.column_names)

# ==========================================
# 5. LoRA CONFIG
# ==========================================
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==========================================
# 6. TRAINING ARGUMENTS
# ==========================================
training_args = TrainingArguments(
    output_dir="./deep_lora_checkpoints",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=20,
    save_steps=20,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    num_train_epochs=3,
    weight_decay=0.1,
    fp16=True,
    gradient_checkpointing=True,
    report_to="none",
    optim="adamw_torch_fused"
)

# ==========================================
# 7. TEST LOSS (LOGGING ONLY)
# ==========================================
def compute_test_loss(model, dataloader):
    model.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            steps += 1

    return total_loss / steps

test_dataloader = torch.utils.data.DataLoader(
    test_tok,
    batch_size=1,
    collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# ==========================================
# 8. CALLBACK (TRAIN LOSS EARLY STOP + CKPT RENAME)
# ==========================================
class TestEarlyStopCallback(TrainerCallback):
    def __init__(self, test_dataloader, patience=3, log_file="train_val_test_log.json"):
        self.test_dataloader = test_dataloader
        self.patience = patience
        self.best_test_loss = float("inf")
        self.counter = 0
        self.log_file = log_file

        self.train_loss_by_step = {}
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.train_loss_by_step[state.global_step] = logs["loss"]

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        step = state.global_step

        # compute TEST loss
        test_loss = compute_test_loss(model, self.test_dataloader)
        train_loss = self.train_loss_by_step.get(step)
        val_loss = metrics.get("eval_loss")

        entry = {
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss
        }
        self.logs.append(entry)

        with open(self.log_file, "w") as f:
            json.dump(self.logs, f, indent=2)

        # ===============================
        # EARLY STOPPING — TEST LOSS ONLY
        # ===============================
        if test_loss < self.best_test_loss:
            self.best_test_loss = test_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("🛑 Early stopping (TEST loss)")
                control.should_training_stop = True

        return control
# ==========================================
# 9. TRAINER
# ==========================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[
        TestEarlyStopCallback(
            test_dataloader=test_dataloader,
            patience=3,
            log_file="deep_train_val_test_log.json"
        )
    ]
)

print("🚀 Eğitim başlıyor...")
trainer.train()

trainer.save_model("./final_deep_model_train_based")

print("✅ Eğitim tamamlandı")
print("📄 Log dosyası: deep_train_val_test_log.json")

