
#DIVERSE TESTING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# 1. CONFIGURATION
base_model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
# Path based on your Drive folder structure
checkpoint_path = "/content/drive/MyDrive/lora_project/models/diverse_instruction/checkpoints/checkpoint-step-540-epoch-0"

print("⏳ Loading Model and Tokenizer...")

# 2. LOAD BASE MODEL & TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa" # Using SDPA to avoid Flash Attention errors
)

# 3. LOAD DIVERSE LORA CHECKPOINT
if os.path.exists(checkpoint_path):
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    print(f"✅ Diverse checkpoint loaded successfully: {checkpoint_path}")
else:
    # Providing a list of found directories to help you debug if the path is slightly different
    parent_dir = os.path.dirname(checkpoint_path)
    if os.path.exists(parent_dir):
        print(f"Directory contents of {parent_dir}: {os.listdir(parent_dir)}")
    raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

# 4. INFERENCE TEST (REQUEST)
prompt = "Calculate the distance between points (x1, y1) and (x2, y2) in three ways: using the standard math library, using NumPy for vectors, and using only basic operators without libraries; then explain the logic for the third approach."
messages = [
    {"role": "system", "content": "You are a helpful coding assistant specialized in diverse programming tasks."},
    {"role": "user", "content": prompt}
]

# Apply chat template for Qwen2.5
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda")

print(f"\nPrompt: {prompt}\n" + "="*50)

# Generate response
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256, 
        do_sample=True, 
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
print(f"Model Response:\n{response}")

#=========/ DEEP TESTING /=========#


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# 1. YOLLARI TANIMLA (Ekran görüntülerine göre ayarlandı)
base_model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
# PDF'deki hiyerarşiye göre Drive yolunu belirtiyoruz
checkpoint_path = "/content/drive/MyDrive/lora_project/models/deep_instruction/checkpoints/checkpoint-step-220-epoch-0"

print("⏳ Model ve Tokenizer yükleniyor...")

# 2. TOKENIZER VE BASE MODELİ YÜKLE
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 3. LORA CHECKPOINT'İNİ YÜKLE
if os.path.exists(checkpoint_path):
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    print(f"✅ LoRA checkpoint başarıyla yüklendi: {checkpoint_path}")
else:
    raise FileNotFoundError(f"Checkpoint bulunamadı: {checkpoint_path}")

# 4. TEST TALEBİ (REQUEST)
prompt = "Calculate the distance between points (x1, y1) and (x2, y2) in three ways: using the standard math library, using NumPy for vectors, and using only basic operators without libraries; then explain the logic for the third approach."
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": prompt}
]

# Chat template uygula
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda")

print(f"\nPrompt: {prompt}\n" + "-"*30)

# Yanıt üret
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256, 
        do_sample=True, 
        temperature=0.7,
        top_p=0.9
    )

response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
print(f"Model Yanıtı:\n{response}")
