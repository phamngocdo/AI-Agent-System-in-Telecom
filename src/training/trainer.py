import os
import torch
import wandb
from dotenv import load_dotenv
from pathlib import Path
from unsloth import FastLanguageModel, get_chat_template
from trl import SFTTrainer, SFTConfig
from transformers import set_seed

load_dotenv()
SEED = 42
set_seed(SEED)
BASE_DIR = Path.cwd()
CACHE_DIR = BASE_DIR.parent / "huggingface_cache"
os.environ["HF_HOME"] = str(CACHE_DIR)

def get_model_and_tokenizer():
    MODEL_NAME = "unsloth/Qwen3-8B-unsloth-bnb-4bit" 
    MAX_SEQ_LENGTH = 2048
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")
    return model, tokenizer