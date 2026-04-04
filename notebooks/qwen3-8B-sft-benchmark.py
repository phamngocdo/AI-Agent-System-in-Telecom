#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
from pathlib import Path

BASE_DIR = Path.cwd()
ROOT_DIR = BASE_DIR.parent
CACHE_DIR = str(ROOT_DIR.parent / "huggingface_cache")

os.environ["HF_HOME"] = CACHE_DIR
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import random
import json
from tqdm import tqdm
import torch
from dotenv import load_dotenv
from transformers import set_seed
from unsloth import FastLanguageModel, get_chat_template
load_dotenv()

SEED = 42
set_seed(SEED)
random.seed(SEED)

BATCH_SIZE = 8

sys.path.insert(0, str(ROOT_DIR))

DATA_DIR = (ROOT_DIR / "data/eval_data/tele-qna.json").resolve()
DATA_OQNA_DIR = (ROOT_DIR / "data/eval_data/tele-eval-10k.json").resolve()

RESULT_MCQ_DIR = (ROOT_DIR / "data/benchmark_results/tele-qna-with-qwen3-tele_2376.json").resolve()
RESULT_OQNA_DIR = (ROOT_DIR / "data/benchmark_results/tele-eval-with-qwen3-tele_2376.json").resolve()

MODEL_DIR = (ROOT_DIR / "models").resolve()


# In[2]:


import unsloth
from unsloth import FastLanguageModel, get_chat_template

def get_model_and_tokenizer():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-8B",
        max_seq_length=2048,
        load_in_4bit=False,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")
    return model, tokenizer


# In[3]:


model, tokenizer = get_model_and_tokenizer()


# In[4]:


model.load_adapter(MODEL_DIR / "sft/checkpoint-2376", adapter_name = "sft_adapter_2376")


# In[5]:


# import json
# import re
# from datasets import Dataset
# from tqdm import tqdm

# with open(DATA_DIR, "r", encoding="utf-8") as f:
#     data = json.load(f)

# dataset = Dataset.from_list(data)


# In[6]:


# def build_texts_mcq(batch):
#     texts = []

#     for sample in batch:
#         choices_text = "\n".join(
#             [f"{i+1}. {c}" for i, c in enumerate(sample["choices"])]
#         )

#         messages = [
#             {
#                 "role": "user",
#                 "content": f"""Quickly answer the following multiple-choice question:

# Question:
# {sample['question']}

# Choices:
# {choices_text}
# """
#             }
#         ]

#         text = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True,
#             enable_thinking=False,
#         )

#         texts.append(text)

#     return texts


# In[7]:


# def generate_batch_mcq(texts):
#     inputs = tokenizer(
#         texts,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#     ).to(model.device)

#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=128,
#         do_sample=False,
#         temperature = 0.7,
#         top_p = 0.8, 
#         top_k = 20,
#     )

#     decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     return decoded


# In[8]:


# import json
# import re
# from tqdm import tqdm

# def extract_after_think(text):
#     parts = re.split(r"</think>\s*\n*", text, maxsplit=1)
#     return parts[1].strip() if len(parts) > 1 else text.strip()

# with open(RESULT_MCQ_DIR, "w", encoding="utf-8") as fout:
#     fout.write("[\n")

#     total = len(dataset)

#     for start in tqdm(range(0, total, BATCH_SIZE)):
#         batch = dataset.select(range(start, min(start + BATCH_SIZE, total)))
#         batch = list(batch)

#         texts = build_texts_mcq(batch)
#         decoded = generate_batch_mcq(texts)

#         for i, sample in enumerate(batch):
#             full_text = decoded[i]
#             clean_text = extract_after_think(full_text) 

#             result = {
#                 "question": sample["question"],
#                 "choices": sample["choices"],
#                 "answer": sample["answer"],
#                 "category": sample.get("category"),
#                 "explaination": sample.get("explaination"),
#                 "model_output": clean_text,
#             }

#             json.dump(result, fout, ensure_ascii=False)

#             global_idx = start + i
#             if global_idx < total - 1:
#                 fout.write(",\n")
#             else:
#                 fout.write("\n")

#         fout.flush()

#     fout.write("]")


# In[10]:


# import json
# import re
# from collections import defaultdict

# import re

# def extract_choice(text):
#     if not text:
#         return None

#     match = re.search(
#         r"(?:The correct answer is|Đáp án đúng là)\s*([A-Da-d]|[1-5])",
#         text
#     )

#     if not match:
#         return None

#     val = match.group(1).upper()

#     if val in ["A", "B", "C", "D"]:
#         return ord(val) - ord("A") + 1
#     return int(val)


# with open((RESULT_MCQ_DIR).resolve(), "r", encoding="utf-8") as f:
#     data = json.load(f)

# total = 0
# correct = 0
# extracted = 0
# failed_indices = []

# cat_stats = defaultdict(lambda: {"total": 0, "correct": 0})
# failed_samples = []  # lưu sample trích xuất thất bại

# for idx, sample in enumerate(data):
#     gt = sample.get("answer")

#     text = sample.get("model_output") or sample.get("think", "")
#     pred = extract_choice(text)

#     total += 1
#     cat = sample.get("category") or "UNKNOWN"
#     cat_stats[cat]["total"] += 1

#     if pred is None:
#         failed_indices.append(idx)
#         failed_samples.append({"index": idx, "text": text})
#     else:
#         extracted += 1
#         if pred == gt:
#             correct += 1
#             cat_stats[cat]["correct"] += 1

# print("\n=== OVERALL ===")
# if total > 0:
#     print(f"Accuracy: {correct}/{total} = {correct/total:.4f}")
# else:
#     print("No data")

# print("\n=== BY CATEGORY ===")
# for cat, stat in cat_stats.items():
#     t = stat["total"]
#     c = stat["correct"]
#     acc = c / t if t > 0 else 0
#     print(f"{cat}: {c}/{t} = {acc:.4f}")

# print("\n=== FAILED EXTRACTIONS ===")
# for item in failed_samples:
#     print(f"Index {item['index']}: {item['text'][:100]}{'...' if len(item['text'])>100 else ''}")


# In[14]:


import json
import re
from datasets import Dataset
from tqdm import tqdm

with open(DATA_OQNA_DIR, "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)


# In[15]:


def build_texts_qna(batch):
    texts = []

    for sample in batch:
        messages = [
            {
                "role": "user",
                "content": f"""Answer this question

Question:
{sample['question']}

Choices:
{sample['answer']}
"""
            }
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        texts.append(text)

    return texts


# In[16]:


def generate_batch_qna(texts):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        temperature = 0.7,
        top_p = 0.8, 
        top_k = 20,
    )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded


# In[17]:


import json
from tqdm import tqdm

MARKER = "\n\nassistant\n<think>\n\n</think>\n\n"

with open(RESULT_OQNA_DIR, "w", encoding="utf-8") as fout:
    fout.write("[\n")

    total = len(dataset)

    for start in tqdm(range(0, total, BATCH_SIZE)):
        batch = dataset.select(range(start, min(start + BATCH_SIZE, total)))
        batch = list(batch)

        texts = build_texts_qna(batch)
        decoded = generate_batch_qna(texts)

        for i, sample in enumerate(batch):
            full_text = decoded[i]

            if MARKER in full_text:
                full_text = full_text.split(MARKER)[1]

            result = {
                "question": sample["question"],
                "answer": sample["answer"],
                "type": sample.get("type"),
                "model_output": full_text,
            }

            json.dump(result, fout, ensure_ascii=False)

            global_idx = start + i
            if global_idx < total - 1:
                fout.write(",\n")
            else:
                fout.write("\n")

        fout.flush()

    fout.write("]")

