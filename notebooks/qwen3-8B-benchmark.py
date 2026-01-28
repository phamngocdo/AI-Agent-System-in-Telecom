#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))

load_dotenv()

HF_CACHE = ROOT.parent / "huggingface_cache"
HF_CACHE.mkdir(exist_ok=True)

os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE)

hf_token = os.getenv("HF_TOKEN")

RESULTS_PATH = ROOT / "data/benchmark_results/qwen3-8B-original"


# In[2]:


import torch
import re
from tqdm import tqdm


# In[3]:


from unsloth import FastLanguageModel, get_chat_template

MODEL_NAME = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-3",
)


# In[4]:


from src.training.trainer import get_model_and_tokenizer
from src.data.qna_loader import QNALoader
from src.data.mcq_loader import MCQLoader

qna_loader = QNALoader(tokenizer=tokenizer)
mcq_loader = MCQLoader(tokenizer=tokenizer)

qna_test = qna_loader.load(splits=("test",), apply_formatting=False)
mcq_test = mcq_loader.load(splits=("test",), apply_formatting=False)


# In[5]:


print(qna_test['test'][0])


# In[6]:


print(mcq_test['test'][0])


# In[7]:


MCQ_SYSTEM_PROMPT = (
    "Bạn là chuyên gia Viễn thông. "
    "CHỈ trả lời bằng SỐ THỨ TỰ của đáp án đúng (1, 2, 3, 4). "
    "KHÔNG giải thích. KHÔNG thêm chữ."
)

mcq_results = []
MAX_NEW_TOKENS_MCQ = 5

for sample in tqdm(mcq_test["test"], desc="MCQ inference"):
    question = sample["question"]
    choices = sample["choices"]

    formatted_choices = ""
    for k in sorted(choices.keys(), key=int):
        formatted_choices += f"{k}. {choices[k]}\n"

    user_content = f"{question}\n\nLựa chọn:\n{formatted_choices}"

    messages = [
        {"role": "system", "content": MCQ_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS_MCQ,
            do_sample=False,
            temperature=0.0,
        )

    gen_text = tokenizer.decode(
        outputs[0][input_ids.shape[-1]:],
        skip_special_tokens=True
    ).strip()

    match = re.findall(r"\d+", gen_text)
    model_answer = int(match[0]) if match else None

    mcq_results.append({
        **sample,
        "model_answer": model_answer,
        "raw_model_output": gen_text,
    })


# In[10]:


QNA_SYSTEM_PROMPT = (
    "Bạn là chuyên gia Viễn thông cao cấp. "
    "Hãy trả lời đúng trọng tâm câu hỏi trong khoảng từ 2 đến 3 câu."
)

qna_results = []

MAX_NEW_TOKENS_QNA = 512
qna_dataset = qna_test["test"]

for sample in tqdm(qna_dataset, desc="QNA inference"):
    question = sample["question"]

    messages = [
        {"role": "system", "content": QNA_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS_QNA,
            do_sample=False,
            temperature=0.0,
        )

    gen_text = tokenizer.decode(
        outputs[0][input_ids.shape[-1]:],
        skip_special_tokens=True
    ).strip()

    qna_results.append({
        **sample,
        "model_answer": gen_text
    })


# In[9]:


import pandas as pd

pd.DataFrame(mcq_results).to_csv(f"{RESULTS_PATH}/mcq_test_predictions.csv", index=False)
pd.DataFrame(qna_results).to_csv(f"{RESULTS_PATH}/qna_test_predictions.csv", index=False)

