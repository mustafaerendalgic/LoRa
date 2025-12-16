# LoRA Fine-Tuning Project  
**Qwen2.5-Coder-1.5B-Instruct**

## Overview

This project focuses on parameter-efficient fine-tuning of a large language model using **LoRA (Low-Rank Adaptation)**.  
The base model **Qwen2.5-Coder-1.5B-Instruct** is fine-tuned on two datasets to improve Python code generation quality while keeping GPU memory usage low.

The project strictly follows the given project specification and includes:
- Training with LoRA
- Loss logging (Train / Validation / Test)
- Required plots (every 20 steps)
- Benchmark evaluation
- Best checkpoint selection

---

## Model

- **Base Model:** Qwen/Qwen2.5-Coder-1.5B-Instruct  
- **Architecture:** Causal Language Model  
- **Precision:** FP16  
- **Fine-Tuning Method:** LoRA  

### LoRA Configuration

| Parameter | Value |
|--------|------|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

---

## Datasets

### CodeGen-Deep-5K (DEEP)
- Algorithmically complex problems
- Focus on deep reasoning and correctness

### CodeGen-Diverse-5K (DIVERSE)
- Broad range of problem types
- Encourages generalization

Each dataset contains:
- `train`
- `valid`
- `test` splits

---

## System Prompt
You are an expert Python programmer.
Please read the problem carefully before writing any Python code.


---

## Training Configuration

| Parameter | Value |
|--------|------|
| Epochs | 3 |
| Batch Size | 2 |
| Gradient Accumulation | 8 |
| Effective Batch Size | 16 |
| Learning Rate | 2e-5 |
| Context Length | 640 |
| Evaluation Interval | Every 20 steps |
| Checkpoint Interval | Every 20 steps |

---

## Loss Logging

Loss values are logged **every 20 steps** as required.

Each CSV file contains:
- Training loss
- Validation loss
- Test loss

### Loss Tables
- `deep_loss_table.csv`
- `diverse_loss_table.csv`

---

## Loss Graphs (Required)

### Deep Dataset Loss Graph
*Train / Validation / Test loss every 20 steps*

![Deep Loss Graph](training_validation_loss_deep.png)

---

### Diverse Dataset Loss Graph
*Train / Validation / Test loss every 20 steps*

![Diverse Loss Graph](training_validation_loss_diverse.png)

---

## Loss Analysis

### Deep Dataset
- Training and validation losses decrease steadily
- Test loss follows validation loss closely
- No strong overfitting observed

### Diverse Dataset
- Faster convergence due to data diversity
- Stable validation behavior
- Better generalization across unseen problems

Overall, both models demonstrate successful learning without memorization.

---

## Benchmark Evaluation

Evaluation is performed using **41 AtCoder problems** via `livecodebench_eval.py`.

Each saved checkpoint is tested, and the best one is selected based on **Pass@1** score.

---

## Best Checkpoint Selection

| Model | Best Checkpoint | Pass@1 | Solved |
|-----|---------------|-------|--------|
| Base Model | – | ?% | 27 / 41 |
| Deep Instruction | step-120 | ?% | 33 / 41 |
| Diverse Instruction | step-180 | ?% | 35 / 41 |

---

## Repository Structure



The following system prompt is **mandatory** and used during training:

