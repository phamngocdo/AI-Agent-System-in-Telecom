#!/usr/bin/env python
# coding: utf-8

# In[63]:


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
load_dotenv()

SEED = 42
set_seed(SEED)
random.seed(SEED)

BATCH_SIZE = 32

sys.path.insert(0, str(ROOT_DIR))

MODEL_DIR = (ROOT_DIR / "models/version3").resolve()
RESULT_DIR = (ROOT_DIR / "data/benchmark_results_1").resolve()


# In[2]:


from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def get_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    return model, tokenizer


# In[3]:


model, tokenizer = get_model_and_tokenizer()


# In[76]:


from datasets import load_dataset

BENCHMARKS = [
    "teleqna",
    "teletables",
    "telemath",
    "telelogs",
    "3gpp_tsg",
    "oranbench",
    "srsranbench",
]

raw_datasets = {
    name: load_dataset("GSMA/ot-full", name, split="test")
    for name in BENCHMARKS
}

# In[77]:


def normalize_dataset(name, ds):
    def _map(example):
        ex = dict(example)
        if name in ["teleqna", "teletables", "oranbench", "srsranbench"]:
            ex["answer"] = ex["answer"] + 1

        return ex

    return ds.map(_map)


# In[79]:


datasets_norm = {}

for name, ds in raw_datasets.items():
    print(f"Normalizing {name} ...")
    datasets_norm[name] = normalize_dataset(name, ds)


# In[54]:


def build_texts_mcq(batch, dataset_name, enable_thinking=False):
    texts = []

    for sample in batch:
        choices = sample["choices"]

        if dataset_name in ["teleqna", "teletables"]:
            choices_text = "\n".join(
                [f"{i+1}. {c}" for i, c in enumerate(choices)]
            )
        else:
            choices_text = "\n".join(choices)

        messages = [
            {
                "role": "user",
                "content": f"""You are an expert in telecommunications. Select the correct answer.

Question:
{sample['question']}

{choices_text}

Respond in the following format:
**The answer is X**. Explanation: your explain

Where X is the option number."""
            }
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        texts.append(text)

    return texts


# In[23]:


MATH_PROMPT = """You are an expert problem solver. Your task is to solve numerical exercises by following these guidelines:
1. **Understand the Goal:** Clearly identify what the problem is asking you to find, paying close attention to the required units for the final answer.
2. **Reason Step-by-Step:** Provide a clear, sequential reasoning process. Explain the formulas, principles, or logic used in each step. Show intermediate calculations if they clarify your thought process. The detailed structure of your sub-steps is up to you, as long as the reasoning is sound and easy to follow.
3. **Unit Management:**
   * Track units throughout your calculations.
   * **Crucially, ensure your final numerical answer is converted to the specific units requested in the problem statement.** If intermediate calculations result in a different unit, perform a final conversion step.
   * State the unit of the final answer clearly in your explanatory text *before* the boxed answer.
4. **Final Numerical Answer Format:**
   * The final answer must be a single numerical value (integer or float).
   * Present this numerical value exclusively within the `\\boxed{{{{...}}}}` format.
   * **CRITICAL:** The `\\boxed{{{{...}}}}` block must contain *only* the number. No text, no units, no labels.

Problem:
{question}
"""

def build_texts_math(batch, enable_thinking=False):
    texts = []

    for sample in batch:
        content = MATH_PROMPT.format(question=sample["question"])

        messages = [
            {"role": "user", "content": content}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        texts.append(text)

    return texts


# In[18]:


def build_texts_cls(batch, enable_thinking=False):
    texts = []

    for sample in batch:
        messages = [
            {
                "role": "user",
                "content": sample["question"]
            }
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        texts.append(text)

    return texts

def build_texts_logs(batch, enable_thinking=False):
    texts = []

    for sample in batch:

        messages = [
            {
                "role": "user",
                "content": f"""
You are a strict classification system.

Task:
Choose EXACTLY ONE correct option from C1 to C8.
run_and_save(
    datasets_norm,
    processed_data_think,
    RESULT_DIR,
    BATCH_SIZE=BATCH_SIZE,
    mode="reasoning",
    think=True
)
IMPORTANT RULES:
- Do NOT explain anything
- Do NOT repeat the question
- Do NOT output text except final answer
- Output format MUST be exactly:

\\boxed{{CX}}

where X ∈ [1,8]

Question:
{sample["question"]}

Return only the final answer.
"""
            }
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        texts.append(text)

    return texts
# In[55]:


def preprocess_dataset(name, dataset, enable_thinking=False):
    batch = list(dataset)

    if name in ["teleqna", "teletables", "oranbench", "srsranbench"]:
        return build_texts_mcq(batch, name, enable_thinking)

    if name == "telemath":
        return build_texts_math(batch, enable_thinking)

    if name in ["telelogs"]:
        return build_texts_logs(batch, enable_thinking)

    if name in ["3gpp_tsg"]:
        return build_texts_cls(batch, enable_thinking)

    raise ValueError(name)


# In[56]:


processed_data_think = {}
processed_data_no_think = {}

for name, ds in raw_datasets.items():
    print(f"Processing {name} ...")

    processed_data_think[name] = preprocess_dataset(
        name,
        ds,
        enable_thinking=True
    )

    processed_data_no_think[name] = preprocess_dataset(
        name,
        ds,
        enable_thinking=False
    )


# In[95]:


def tokenize_all(texts):
    return tokenizer(
        texts,
        padding=False,
        truncation=True
    )


# In[96]:


def generate_from_tokenized(tokenized, start, end):
    batch = {
        "input_ids": tokenized["input_ids"][start:end],
        "attention_mask": tokenized["attention_mask"][start:end],
    }

    batch = tokenizer.pad(
        batch,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **batch,
        max_new_tokens=8192,
        do_sample=True,
        return_dict_in_generate=True,
    )

    return tokenizer.batch_decode(
        outputs.sequences,
        skip_special_tokens=True
    )


# In[97]:


import os
import json
import time
from tqdm import tqdm

def run_and_save(
    raw_datasets,
    processed_data,
    RESULT_DIR,
    BATCH_SIZE=8,
    mode="general",
    think=False
):
    save_dir = os.path.join(RESULT_DIR, "think" if think else "non_think")
    os.makedirs(save_dir, exist_ok=True)

    for name in datasets_norm:
        print(f"\n=== Running {name} ===")

        raw_ds = datasets_norm[name]
        texts = processed_data[name]

        tokenized = tokenize_all(texts)

        save_path = os.path.join(save_dir, f"telcollm-{name}.jsonl")

        done = 0
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                done = sum(1 for _ in f)
            print(f"Resume from {done} samples")

        with open(save_path, "a", encoding="utf-8") as f:

            for i in tqdm(range(done, len(texts), BATCH_SIZE)):
                batch_raw = [
                    raw_ds[k] for k in range(i, min(i+BATCH_SIZE, len(raw_ds)))
                ]

                outputs = generate_from_tokenized(
                    tokenized,
                    i,
                    i + BATCH_SIZE
                )

                for j in range(len(outputs)):
                    result = dict(batch_raw[j])
                    result["model_output"] = outputs[j]

                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"Saved → {save_path}")


# In[85]:


model.generation_config.max_length = None


# In[100]:


run_and_save(
    datasets_norm,
    processed_data_no_think,
    RESULT_DIR,
    BATCH_SIZE=BATCH_SIZE,
    mode="general",
    think=False
)


# In[ ]:


run_and_save(
    datasets_norm,
    processed_data_think,
    RESULT_DIR,
    BATCH_SIZE=BATCH_SIZE,
    mode="reasoning",
    think=True
)

