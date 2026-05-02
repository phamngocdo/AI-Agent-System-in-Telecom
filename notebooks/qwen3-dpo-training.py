#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

MAX_SEQ_LENGTH = 4096

sys.path.insert(0, str(ROOT_DIR))

DATA_UN_LABEL_DIR = (ROOT_DIR / "data/dpo_data/un_label").resolve()
DATA_LABELED_DIR = (ROOT_DIR / "data/dpo_data/labeled/data.json").resolve()
RESULT_DIR = (ROOT_DIR / "data/dpo_data/result_for_unlabel").resolve()
MODEL_DIR = (ROOT_DIR / "models").resolve()


# In[2]:


import unsloth
from unsloth import FastLanguageModel, get_chat_template, PatchDPOTrainer

def get_model_and_tokenizer():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-8B",
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        device_map={"": local_rank},
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")
    return model, tokenizer


# In[3]:


model, tokenizer = get_model_and_tokenizer()

from peft import PeftModel

model = PeftModel.from_pretrained(model, MODEL_DIR / "continual-pretrain", adapter_name="pretrain")
model.merge_adapter(["pretrain"])

model.load_adapter(MODEL_DIR / "sft", adapter_name="sft")
model.merge_adapter(["sft"])

# In[9]:

def preprocess_function(example):
    return {
        "prompt": [
            {"role": "user", "content": example["prompt"]}
        ],
        "chosen": [
            {"role": "assistant", "content": example["chosen"]}
        ],
        "rejected": [
            {"role": "assistant", "content": example["rejected"]}
        ],
    }


# In[21]:


from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files=str(DATA_LABELED_DIR)
)


# In[22]:


dataset = dataset.map(preprocess_function)


# In[23]:


split_dataset = dataset["train"].train_test_split(
    test_size=0.1,
    seed=42
)

train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]


# In[8]:


model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = SEED,
    use_rslora = True,
    loftq_config = None,
)


# In[27]:


from unsloth import is_bfloat16_supported
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=64,

    warmup_ratio=0.1,
    num_train_epochs=1,
    learning_rate=5e-6,

    logging_steps=1,
    eval_strategy="steps",
    eval_steps=1,

    save_strategy="steps",
    save_steps=1,
    save_total_limit=3,

    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=SEED,

    output_dir=MODEL_DIR / "dpo",

    report_to="wandb",

    gradient_checkpointing=False,
)


# In[28]:


trainer = DPOTrainer(
    model=model,
    ref_model=None,
    tokenizer=tokenizer,

    train_dataset=train_dataset,
    eval_dataset=eval_dataset,

    args=dpo_config,
)


# In[33]:


trainer_stats = trainer.train()

