import os
import gc
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)
from peft import PeftModel


# =====================================================
# CONFIGURATION
# =====================================================

BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

DATASET_PATH = "CodeGen-Deep-5K.jsonl"
CHECKPOINT_DIR = "models/deep_instruction/checkpoints"
OUTPUT_FILE = "results/deep_test_loss_results.json"

MAX_LENGTH = 1024
BATCH_SIZE = 4


# =====================================================
# DEVICE
# =====================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")


# =====================================================
# TOKENIZER
# =====================================================

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=False)
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# =====================================================
# DATASET PREPARATION (TRAIN-CONSISTENT)
# =====================================================

def preprocess(example):
    system_prompt = (
        "You are an expert Python programmer. "
        "Please read the problem carefully before writing any Python code.\n"
    )
    example["text"] = system_prompt + example["solution"] + tokenizer.eos_token
    return example


def tokenize(example):
    out = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )
    out["labels"] = out["input_ids"].copy()
    return out


print("Loading dataset...")
raw = load_dataset("json", data_files=DATASET_PATH, split="train")

test_ds = raw.filter(lambda x: x.get("split") == "test")
test_ds = test_ds.map(preprocess)
test_tok = test_ds.map(
    tokenize,
    batched=True,
    remove_columns=test_ds.column_names
)

print(f"Test samples: {len(test_ds)}")

eval_dataloader = torch.utils.data.DataLoader(
    test_tok,
    batch_size=BATCH_SIZE,
    collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)


# =====================================================
# LOSS COMPUTATION
# =====================================================

def calculate_loss(model, dataloader):
    model.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            steps += 1

    return total_loss / steps if steps > 0 else 0.0


# =====================================================
# EVALUATION LOOP
# =====================================================

results = []

if not os.path.exists(CHECKPOINT_DIR):
    raise FileNotFoundError(f"Checkpoint directory not found: {CHECKPOINT_DIR}")

checkpoints = [
    d for d in os.listdir(CHECKPOINT_DIR)
    if d.startswith("checkpoint")
]

checkpoints.sort(key=lambda x: int(x.split("-")[2]))

print(f"Found {len(checkpoints)} checkpoints")

for ckpt in checkpoints:
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt)
    print(f"Evaluating {ckpt}")

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        base_model.config.use_cache = False

        model = PeftModel.from_pretrained(base_model, ckpt_path)

        loss = calculate_loss(model, eval_dataloader)

        results.append({
            "checkpoint": ckpt,
            "step": int(ckpt.split("-")[2]),
            "test_loss": loss
        })

        print(f"  Test loss: {loss:.4f}")

    except Exception as e:
        print(f"  Error evaluating {ckpt}: {e}")

    finally:
        del model
        del base_model
        gc.collect()
        torch.cuda.empty_cache()


# =====================================================
# SAVE RESULTS
# =====================================================

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print("Evaluation completed.")
print(f"Results saved to {OUTPUT_FILE}")


###--------------------- Diverse

import os
import gc
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)
from peft import PeftModel

%cd /content

# =====================================================
# CONFIGURATION
# =====================================================

BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

DATASET_PATH = "CodeGen-Deep-5K.jsonl"
CHECKPOINT_DIR = "/content/CodeGen/models/diverse_instruction/checkpoints"
OUTPUT_FILE = "results/diverse_test_loss_results.json"

MAX_LENGTH = 1024
BATCH_SIZE = 4


# =====================================================
# DEVICE
# =====================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")


# =====================================================
# TOKENIZER
# =====================================================

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=False)
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# =====================================================
# DATASET PREPARATION (TRAIN-CONSISTENT)
# =====================================================

def preprocess(example):
    system_prompt = (
        "You are an expert Python programmer. "
        "Please read the problem carefully before writing any Python code.\n"
    )
    example["text"] = system_prompt + example["solution"] + tokenizer.eos_token
    return example


def tokenize(example):
    out = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )
    out["labels"] = out["input_ids"].copy()
    return out


print("Loading dataset...")
raw = load_dataset("json", data_files=DATASET_PATH, split="train")

test_ds = raw.filter(lambda x: x.get("split") == "test")
test_ds = test_ds.map(preprocess)
test_tok = test_ds.map(
    tokenize,
    batched=True,
    remove_columns=test_ds.column_names
)

print(f"Test samples: {len(test_ds)}")

eval_dataloader = torch.utils.data.DataLoader(
    test_tok,
    batch_size=BATCH_SIZE,
    collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)


# =====================================================
# LOSS COMPUTATION
# =====================================================

def calculate_loss(model, dataloader):
    model.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            steps += 1

    return total_loss / steps if steps > 0 else 0.0


# =====================================================
# EVALUATION LOOP
# =====================================================

results = []

if not os.path.exists(CHECKPOINT_DIR):
    raise FileNotFoundError(f"Checkpoint directory not found: {CHECKPOINT_DIR}")

checkpoints = [
    d for d in os.listdir(CHECKPOINT_DIR)
    if d.startswith("checkpoint")
]

checkpoints.sort(key=lambda x: int(x.split("-")[2]))

print(f"Found {len(checkpoints)} checkpoints")

for ckpt in checkpoints:
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt)
    print(f"Evaluating {ckpt}")

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        base_model.config.use_cache = False

        model = PeftModel.from_pretrained(base_model, ckpt_path)

        loss = calculate_loss(model, eval_dataloader)

        results.append({
            "checkpoint": ckpt,
            "step": int(ckpt.split("-")[2]),
            "test_loss": loss
        })

        print(f"  Test loss: {loss:.4f}")

    except Exception as e:
        print(f"  Error evaluating {ckpt}: {e}")

    finally:
        del model
        del base_model
        gc.collect()
        torch.cuda.empty_cache()


# =====================================================
# SAVE RESULTS
# =====================================================

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print("Evaluation completed.")
print(f"Results saved to {OUTPUT_FILE}")

