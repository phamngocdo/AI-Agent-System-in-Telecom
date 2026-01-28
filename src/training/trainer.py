import os
import torch
import wandb
from pathlib import Path
from dotenv import load_dotenv
from datasets import concatenate_datasets
from transformers import set_seed
from unsloth import FastLanguageModel, get_chat_template
from trl import SFTTrainer, SFTConfig

from src.data.qna_loader import QNALoader
from src.data.mcq_loader import MCQLoader

load_dotenv()
SEED = 42
set_seed(SEED)
BASE_DIR = Path.cwd()
CACHE_DIR = BASE_DIR.parent / "huggingface_cache"
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_model_and_tokenizer(config):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")
    return model, tokenizer

def train():
    run = wandb.init()
    config = wandb.config

    model, tokenizer = get_model_and_tokenizer(config)

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config.lora_alpha,
        lora_dropout=0,
        bias="none",
    )

    qna_loader = QNALoader(tokenizer=tokenizer)
    mcq_loader = MCQLoader(tokenizer=tokenizer)

    qna_full = qna_loader.load(splits=["train"])["train"].select(range(10))
    mcq_full = mcq_loader.load(splits=["train"])["train"].select(range(10))

    qna_split = qna_full.train_test_split(test_size=0.1, stratify_by_column="category", seed=config.seed)
    mcq_split = mcq_full.train_test_split(test_size=0.1, stratify_by_column="category", seed=config.seed)

    train_ds = concatenate_datasets([qna_split["train"], mcq_split["train"]]).shuffle(seed=config.seed)
    val_ds = concatenate_datasets([qna_split["test"], mcq_split["test"]]).shuffle(seed=config.seed)

    training_args = SFTConfig(
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=config.logging_steps,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        optim="adamw_8bit",
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        seed=config.seed,
        output_dir="models",
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        args=training_args,
    )

    trainer.train()
    wandb.finish()

if __name__ == "__main__":
    train()