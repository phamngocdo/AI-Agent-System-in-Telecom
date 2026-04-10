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

SEED = 36
set_seed(SEED)
random.seed(SEED)

BATCH_SIZE = 8

sys.path.insert(0, str(ROOT_DIR))

DATA_DIR = (ROOT_DIR / "data/sft_data/train.json").resolve()
MODEL_DIR = (ROOT_DIR / "models").resolve()


# In[4]:


import wandb

if os.getenv("WANDB_API_KEY"):
    wandb.login(key=os.getenv("WANDB_API_KEY"))
else:
    print("WANDB_API_KEY not found")
    
wandb.init(
    project="Qwen3-sft-training",
)

# In[2]:


import unsloth
from unsloth import FastLanguageModel, get_chat_template

def get_model_and_tokenizer():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-8B",
        max_seq_length=2048,
        load_in_4bit=True,
        device_map = "balanced"
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")
    return model, tokenizer

# In[3]:


model, tokenizer = get_model_and_tokenizer()
model.load_adapter(MODEL_DIR / "continual-pretrain")

# In[6]:


from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files=str(DATA_DIR),
    split="train"
).shuffle(seed=SEED)

dataset


# In[9]:


def generate_conversation(examples):
    problems  = examples["instruction"]
    solutions = examples["output"]
    conversations = []
    for problem, solution in zip(problems, solutions):
        conversations.append([
            {"role" : "user",      "content" : problem},
            {"role" : "assistant", "content" : solution},
        ])
    return { "conversations": conversations, }


# In[12]:


dataset = dataset.map(generate_conversation, batched=True)

def format_chat(batch):
    texts = tokenizer.apply_chat_template(
        batch["conversations"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )

    return {"text": texts}

dataset = dataset.map(format_chat, batched=True)
dataset = dataset.remove_columns(["instruction", "output", "conversations"])

# In[15]:

dataset = dataset.train_test_split(test_size=0.05, seed=SEED)


# In[16]:


model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = SEED,
    use_rslora = False,
    loftq_config = None,
)


# In[18]:


from transformers import TrainerCallback
from unsloth import UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only


class SkipNaNCallback(TrainerCallback):
    def __init__(self, tokenizer, save_dir):
        self.tokenizer = tokenizer
        self.save_dir = Path(save_dir)
        self._last_inputs = None

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        loss = logs.get("loss") if logs else None
        if loss is None:
            return
        if not torch.isfinite(torch.tensor(float(loss))):
            step = state.global_step
            print(f"[SkipNaNCallback] NaN/Inf at step {step} | loss={loss}")
            inputs = self._last_inputs
            if inputs is not None and "input_ids" in inputs:
                for i, ids in enumerate(inputs["input_ids"]):
                    text = self.tokenizer.decode(ids.tolist(), skip_special_tokens=False)
                    print(f"  [Bad sample {i}] length={len(ids)} | text[:500]={text[:500]!r}")
            if model is not None:
                ckpt_dir = self.save_dir / f"nan_step_{step}"
                model.save_pretrained(str(ckpt_dir))
                self.tokenizer.save_pretrained(str(ckpt_dir))
                print(f"[SkipNaNCallback] LoRA saved to {ckpt_dir}")


nan_callback = SkipNaNCallback(tokenizer, save_dir=MODEL_DIR / "sft1")




training_args = UnslothTrainingArguments(
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,

    warmup_ratio=0.1,
    num_train_epochs=3,
    learning_rate=2e-5,
    max_grad_norm=1.0,

    eval_strategy="steps",
    eval_steps=100,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=3,

    report_to="wandb",

    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=SEED,

    output_dir=MODEL_DIR / "sft",
    gradient_checkpointing=False,
)

trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=8,
    packing=False,
    args=training_args,
    callbacks=[nan_callback],
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

_orig_prepare = trainer._prepare_inputs
def _patched_prepare(inputs):
    prepared = _orig_prepare(inputs)
    nan_callback._last_inputs = prepared
    return prepared
trainer._prepare_inputs = _patched_prepare


# In[19]: Verify masking
tokenizer.decode(trainer.train_dataset[10]["input_ids"])
tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " ")


# In[20]:
trainer.train()
trainer.evaluate()



