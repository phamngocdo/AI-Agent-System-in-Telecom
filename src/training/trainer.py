from unsloth import FastLanguageModel, get_chat_template
import torch

model_name = "unsloth/Qwen3-8B-unsloth-bnb-4bit" 

_, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048,
    dtype = torch.float16,
    load_in_4bit = True,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-3",
)