#DEEP TRAINING
import torch
import os
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model


# =====================
# MODEL & TOKENIZER
# =====================
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Gradient Checkpointing (OOM fix)
model.gradient_checkpointing_enable()
model.config.use_cache = False


# =====================
# DATASET (DEEP)
# =====================
dataset = load_dataset("Naholav/CodeGen-Deep-5K")["train"]

SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Please read the problem carefully before writing any Python code."
)

def format_example(example):
    return {
        "text": (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|assistant|>\n{example['solution']}\n"
        ),
        "split": example["split"]
    }

dataset = dataset.map(format_example)

train_text = dataset.filter(lambda x: x["split"] == "train")
valid_text = dataset.filter(lambda x: x["split"] == "valid")
test_text  = dataset.filter(lambda x: x["split"] == "test")


# =====================
# TOKENIZATION
# =====================
MAX_LEN = 800

def tokenize(example):
    out = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length"
    )
    out["labels"] = out["input_ids"].copy()
    return out

train_ds = train_text.map(tokenize, remove_columns=train_text.column_names)
valid_ds = valid_text.map(tokenize, remove_columns=valid_text.column_names)
test_ds  = test_text.map(tokenize, remove_columns=test_text.column_names)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


# =====================
# LoRA CONFIG
# =====================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

model = get_peft_model(model, lora_config)


# =====================
# CHECKPOINT NAMING CALLBACK
# =====================
class RenameCheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.epoch is None:
            return

        step = state.global_step
        epoch = int(state.epoch)

        old_path = os.path.join(args.output_dir, f"checkpoint-{step}")
        new_path = os.path.join(
            args.output_dir,
            f"checkpoint-step-{step}-epoch-{epoch}"
        )

        if os.path.exists(old_path) and not os.path.exists(new_path):
            os.rename(old_path, new_path)


# =====================
# TRAINING ARGUMENTS
# =====================
training_args = TrainingArguments(
    output_dir="models/deep_instruction/checkpoints",

    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=1,

    num_train_epochs=3,
    learning_rate=2e-5,

    logging_steps=20,
    eval_strategy="steps",
    eval_steps=20,
    save_steps=20,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    max_grad_norm=1.0,

    fp16=True,
    report_to="none"
)


# =====================
# TRAINER
# =====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=data_collator,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=2),
        RenameCheckpointCallback()
    ]
)


# =====================
# TRAIN
# =====================
trainer.train()


# =====================
# TEST EVALUATION
# =====================
test_metrics = trainer.evaluate(test_ds)
test_loss = test_metrics["eval_loss"]


# =====================
# LOSS GRAPH (REPORT)
# =====================
train_losses = []
val_losses = []

for log in trainer.state.log_history:
    if "loss" in log:
        train_losses.append(log["loss"])
    if "eval_loss" in log:
        val_losses.append(log["eval_loss"])

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.axhline(test_loss, linestyle="--", label="Test Loss")
plt.legend()
plt.xlabel("Steps (x20)")
plt.ylabel("Loss")
plt.title("Train / Validation / Test Loss")
plt.show()

#DIVERSE TRAINING --------------------------------------------------------------------------------

import torch
import os
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model


# =====================
# MODEL & TOKENIZER
# =====================
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Gradient checkpointing (OOM fix)
model.gradient_checkpointing_enable()
model.config.use_cache = False


# =====================
# DATASET (DIVERSE)
# =====================
dataset = load_dataset("Naholav/CodeGen-Diverse-5K")["train"]

SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Please read the problem carefully before writing any Python code."
)

def format_example(example):
    return {
        "text": (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|assistant|>\n{example['solution']}\n"
        ),
        "split": example["split"]
    }

dataset = dataset.map(format_example)

train_text = dataset.filter(lambda x: x["split"] == "train")
valid_text = dataset.filter(lambda x: x["split"] == "valid")
test_text  = dataset.filter(lambda x: x["split"] == "test")


# =====================
# TOKENIZATION
# =====================
MAX_LEN = 800

def tokenize(example):
    out = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length"
    )
    out["labels"] = out["input_ids"].copy()
    return out

train_ds = train_text.map(tokenize, remove_columns=train_text.column_names)
valid_ds = valid_text.map(tokenize, remove_columns=valid_text.column_names)
test_ds  = test_text.map(tokenize, remove_columns=test_text.column_names)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


# =====================
# LoRA CONFIG
# =====================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

model = get_peft_model(model, lora_config)


# =====================
# CHECKPOINT NAMING CALLBACK
# =====================
class RenameCheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.epoch is None:
            return

        step = state.global_step
        epoch = int(state.epoch)

        old_path = os.path.join(args.output_dir, f"checkpoint-{step}")
        new_path = os.path.join(
            args.output_dir,
            f"checkpoint-step-{step}-epoch-{epoch}"
        )

        if os.path.exists(old_path) and not os.path.exists(new_path):
            os.rename(old_path, new_path)


# =====================
# TRAINING ARGUMENTS
# =====================
training_args = TrainingArguments(
    output_dir="models/diverse_instruction/checkpoints",

    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=1,

    num_train_epochs=3,
    learning_rate=2e-5,

    logging_steps=20,
    eval_strategy="steps",
    eval_steps=20,
    save_steps=20,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    max_grad_norm=1.0,

    fp16=True,
    report_to="none"
)


# =====================
# TRAINER
# =====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=data_collator,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=2),
        RenameCheckpointCallback()
    ]
)


# =====================
# TRAIN
# =====================
trainer.train()


# =====================
# TEST EVALUATION
# =====================
test_metrics = trainer.evaluate(test_ds)
test_loss = test_metrics["eval_loss"]


# =====================
# LOSS GRAPH (REPORT)
# =====================
train_losses = []
val_losses = []

for log in trainer.state.log_history:
    if "loss" in log:
        train_losses.append(log["loss"])
    if "eval_loss" in log:
        val_losses.append(log["eval_loss"])

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.axhline(test_loss, linestyle="--", label="Test Loss")
plt.legend()
plt.xlabel("Steps (x20)")
plt.ylabel("Loss")
plt.title("DIVERSE: Train / Validation / Test Loss")
plt.show()