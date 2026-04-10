#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys, torch
from pathlib import Path
from dotenv import load_dotenv
import unsloth
from transformers import set_seed

load_dotenv()

SEED = 42
set_seed(SEED)

BASE_DIR = Path.cwd()
ROOT_DIR = BASE_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

CACHE_DIR = ROOT_DIR.parent / "huggingface_cache"
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_DIR = (ROOT_DIR / "data/pretrain_data/processed").resolve()
MODEL_DIR = (ROOT_DIR / "models/").resolve()


# In[2]:


import wandb

if os.getenv("WANDB_API_KEY"):
    wandb.login(key=os.getenv("WANDB_API_KEY"))
else:
    print("WANDB_API_KEY not found")

wandb.init(
    project="Qwen3-Pretraining",
)

# In[3]:


from unsloth import FastLanguageModel, get_chat_template

def get_model_and_tokenizer():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-8B",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")
    return model, tokenizer


# In[4]:


model, tokenizer = get_model_and_tokenizer()


# In[ ]:

from datasets import load_dataset, concatenate_datasets, DatasetDict

PROMPT_3GPP = """3GPP Standard
### Title: {title}
### Text: {text}"""

PROMPT_ARXIV = """Arxiv
### Title: {title}
### Text: {text}"""

PROMPT_BOOKS = """Books
### Title: {title}
### Text: {text}"""

PROMPT_MAP = {
    "3gpp": PROMPT_3GPP,
    "arxiv": PROMPT_ARXIV,
    "books": PROMPT_BOOKS,
}

MAX_SEQ_LENGTH = 2048
CHUNK_STRIDE   = 256

def _chunk_batch(batch, tokenizer):
    all_texts, all_domains = [], []
    EOS = tokenizer.eos_token
    for title, raw_text, domain in zip(batch["title"], batch["raw_text"], batch["domain"]):
        PROMPT = PROMPT_MAP.get(domain, PROMPT_ARXIV)
        tokenized = tokenizer(
            raw_text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_overflowing_tokens=True,
            stride=CHUNK_STRIDE,
            add_special_tokens=False,
        )
        for ids in tokenized["input_ids"]:
            chunk_body = tokenizer.decode(ids, skip_special_tokens=True)
            all_texts.append(PROMPT.format(title=title, text=chunk_body) + EOS)
            all_domains.append(domain)
    return {"text": all_texts, "domain": all_domains}


def load_all_pretrain_data(root_dir: str, tokenizer, seed=42):
    root = Path(root_dir)
    datasets_list = []

    for file in (root / "3GPP").glob("*.jsonl"):
        release_name = file.stem.replace("Rel-", "Release ")
        ds = load_dataset("json", data_files=str(file), split="train")
        ds = ds.map(
            lambda x: {"title": release_name, "raw_text": x["text"], "domain": "3gpp"},
            remove_columns=ds.column_names,
        )
        datasets_list.append(ds)

    rp_file = next((root / "redpajama").glob("*.jsonl"))
    ds_rp = load_dataset("json", data_files=str(rp_file), split="train")
    ds_rp = ds_rp.map(
        lambda x: {"title": x["file_name"], "raw_text": x["text"], "domain": "arxiv"},
        remove_columns=ds_rp.column_names,
    )
    datasets_list.append(ds_rp)

    book_file = next((root / "books").glob("*.jsonl"))
    ds_book = load_dataset("json", data_files=str(book_file), split="train")
    ds_book = ds_book.map(
        lambda x: {"title": x["file_name"], "raw_text": x["text"], "domain": "arxiv"},
        remove_columns=ds_book.column_names,
    )
    datasets_list.append(ds_book)

    full_dataset = concatenate_datasets(datasets_list)

    full_dataset = full_dataset.map(
        lambda batch: _chunk_batch(batch, tokenizer),
        batched=True,
        batch_size=64,
        remove_columns=full_dataset.column_names,
        num_proc=1,
        desc="Chunking documents",
    )

    full_dataset = full_dataset.shuffle(seed=seed)
    return full_dataset


# In[ ]:


from datasets import ClassLabel

def stratified_split(dataset, val_ratio=0.05, seed=42):
    from datasets import ClassLabel, DatasetDict

    domain_names = sorted(set(dataset["domain"]))
    dataset = dataset.cast_column("domain", ClassLabel(names=domain_names))

    split_dataset = dataset.train_test_split(
        test_size=val_ratio,
        seed=seed,
        stratify_by_column="domain"
    )

    return DatasetDict({
        "train": split_dataset["train"],
        "validation": split_dataset["test"]
    })


# In[ ]:


dataset = load_all_pretrain_data(DATA_DIR, tokenizer)
dataset = stratified_split(dataset, val_ratio=0.1)

print(dataset)


# In[ ]:


# import pandas as pd

# def count_tokens(example):
#     return {"length": len(tokenizer(example["text"])["input_ids"])}

# def get_stats(split):
#     length_ds = dataset[split].map(count_tokens, num_proc=4)
#     df = pd.DataFrame({
#         "domain": length_ds["domain"],
#         "tokens": length_ds["length"],
#     })
#     stats = df.groupby("domain")["tokens"].agg(
#         ["count", "sum", "mean", "max"]
#     ).reset_index()
#     stats.columns = [
#         "domain",
#         f"{split}_num_samples",
#         f"{split}_total_tokens",
#         f"{split}_avg_tokens",
#         f"{split}_max_tokens",
#     ]
#     return stats

# train_stats = get_stats("train")
# val_stats   = get_stats("validation")

# final_stats = train_stats.merge(val_stats, on="domain", how="outer")

# final_stats["overall_total_tokens"] = (
#     final_stats["train_total_tokens"] +
#     final_stats["validation_total_tokens"]
# )

# final_stats["val_token_ratio"] = (
#     final_stats["validation_total_tokens"] /
#     final_stats["overall_total_tokens"]
# )

# final_stats.sort_values("domain")


# In[ ]:


model = FastLanguageModel.get_peft_model(
    model,
    r = 128,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = SEED,
    use_rslora = True,
    loftq_config = None,
)


# In[ ]:


from unsloth import UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported


trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset["validation"],
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 8,

    packing = False,

    args = UnslothTrainingArguments(
        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 2,
        gradient_accumulation_steps = 4,

        warmup_ratio = 0.1,
        num_train_epochs = 1,
        learning_rate = 5e-5,

        logging_steps = 10,
        eval_strategy = "steps",
        eval_steps = 300,

        save_strategy = "steps",
        save_steps = 300,
        save_total_limit = 3,
        load_best_model_at_end = True,

        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = SEED,

        output_dir = MODEL_DIR,
        report_to = "wandb",

        gradient_checkpointing = True,
    ),
)


# In[ ]:


trainer_stats = trainer.train()
trainer.evaluate()
