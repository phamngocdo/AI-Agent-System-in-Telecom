#!/usr/bin/env python
# coding: utf-8

# In[5]:


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

DATA_MCQ_DIR = (ROOT_DIR / "data/eval_data/tele-qna.json").resolve()
DATA_OQNA_DIR = (ROOT_DIR / "data/eval_data/tele-eval-10k.json").resolve()
RESULT_MCQ_DIR = (ROOT_DIR / "data/benchmark_results/tele-qna-with-qwen3-original.json").resolve()
RESULT_OQNA_DIR = (ROOT_DIR / "data/benchmark_results/tele-eval-with-qwen3-original.json").resolve()
MODEL_DIR = (ROOT_DIR / "models").resolve()


# In[3]:


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


# In[4]:


model, tokenizer = get_model_and_tokenizer()


## In[4]:


# import json
# import re
# from datasets import Dataset
# from tqdm import tqdm

# with open(DATA_MCQ_DIR, "r", encoding="utf-8") as f:
#     data = json.load(f)

# dataset = Dataset.from_list(data)


## In[5]:


# def build_texts_mcq(batch):
#     texts = []

#     for sample in batch:
#         choices_text = "\n".join(
#             [f"{i+1}. {c}" for i, c in enumerate(sample["choices"])]
#         )

#         messages = [
#             {
#                 "role": "user",
#                 "content": f"""You are a multiple choice QA system.

# Select exactly ONE correct answer.

# Output format (STRICT):
# Đáp án là X

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


# # In[6]:


# def generate_batch_mcq(texts):
#     inputs = tokenizer(
#         texts,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#     ).to(model.device)

#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=10,
#         do_sample=False,
#         temperature = 0.7,
#         top_p = 0.8, 
#         top_k = 20,
#     )

#     decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     return decoded


# In[7]:


# import json
# from tqdm import tqdm

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

#             result = {
#                 "question": sample["question"],
#                 "choices": sample["choices"],
#                 "answer": sample["answer"],
#                 "category": sample.get("category"),
#                 "explaination": sample.get("explaination"),
#                 "model_output": answer_text,
#             }

#             json.dump(result, fout, ensure_ascii=False)

#             global_idx = start + i
#             if global_idx < total - 1:
#                 fout.write(",\n")
#             else:
#                 fout.write("\n")

#         fout.flush()

#     fout.write("]")


# In[5]:


# import json
# import re
# from collections import defaultdict

# def extract_choice(text):
#     if not text:
#         return None
#     match = re.search(r"\b([1-4])\b", text)
#     return int(match.group(1)) if match else None

# with open((ROOT_DIR / "data/benchmark_results/tele-qna-with-qwen3-original.json").resolve(), "r", encoding="utf-8") as f:
#     data = json.load(f)



# total = 0
# correct = 0

# cat_stats = defaultdict(lambda: {"total": 0, "correct": 0})

# for idx, sample in enumerate(data):
#     gt = sample.get("answer")

#     text = sample.get("model_output") or sample.get("think", "")
#     pred = extract_choice(text)

#     total += 1
#     cat = sample.get("category") or "UNKNOWN"

#     cat_stats[cat]["total"] += 1

#     if pred == gt:
#         correct += 1
#         cat_stats[cat]["correct"] += 1


# print("=== OVERALL ===")
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


# In[6]:


import json
import re
from datasets import Dataset
from tqdm import tqdm

with open(DATA_OQNA_DIR, "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)


# In[8]:


def build_texts_qna(batch):
    texts = []

    for sample in batch:
        messages = [
            {
                "role": "user",
                "content": f"""Answer the following question clearly and concisely.

Question:
{sample['question']}

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


# In[9]:


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


# In[11]:


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

