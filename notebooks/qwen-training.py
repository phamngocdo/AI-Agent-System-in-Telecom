#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
print(Path.cwd())
parent_dir = Path.cwd().parent.parent

hf_token = os.getenv("HF_TOKEN")
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
os.environ["HF_HOME"] = str(parent_dir / "huggingface_cache")


# In[2]:


import wandb

wandb.init(
    project="Qwen3-8B-Telecom-Training",
)
config = wandb.config


# In[3]:


from transformers import set_seed

set_seed(42)


# In[4]:


from unsloth import FastLanguageModel, get_chat_template
import torch

model_name = "unsloth/Qwen3-8B-unsloth-bnb-4bit" 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048,
    dtype = torch.float16,
    load_in_4bit = True,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-3",
)


# In[5]:


def formatting_prompts_func(examples):
    questions    = examples["question"]
    choices_list = examples["choices"]
    answers      = examples["answer"]
    explanations = examples["explanation"]
    
    texts = []
    system_prompt = (
        "Bạn là chuyên gia Viễn thông cao cấp. "
        "Hãy phân tích câu hỏi trắc nghiệm, thực hiện suy luận logic trong thẻ <think> "
        "để chọn phương án đúng và giải thích lý do."
    )

    for q, choices, ans_idx, expl in zip(questions, choices_list, answers, explanations):
        formatted_choices = ""
        for i, choice_text in enumerate(choices):
            formatted_choices += f"{i + 1}. {choice_text}\n"
        
        user_content = f"{q}\n\nLựa chọn:\n{formatted_choices}"
        
        try:
            correct_choice_text = choices[int(ans_idx) - 1]
            response_label = f"Phương án {ans_idx}"
        except (IndexError, ValueError):
            correct_choice_text = "Không xác định"
            response_label = "N/A"
        
        assistant_content = f"<think>\n{expl}\n</think>\nĐáp án đúng là: {response_label}. {correct_choice_text}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = False)
        texts.append(text)
        
    return { "text" : texts, }


# In[6]:


from datasets import load_dataset

raw_dataset = load_dataset("phamngocdo/Vietnamese_TeleMCQ_dataset")["train"].shuffle(seed=42)

train_test_split = raw_dataset.train_test_split(test_size=0.05)
test_valid_split = train_test_split["test"]

dataset = {
    "train": train_test_split["train"],
    "validation": test_valid_split,
}

dataset["train"] = dataset["train"].map(formatting_prompts_func, batched = True)
dataset["validation"] = dataset["validation"].map(formatting_prompts_func, batched = True)

print(f"Train size: {len(dataset["train"])}")
print(f"Validation size: {len(dataset["validation"])}")
print(dataset["train"][0]["text"])


# In[7]:


model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 32,
    lora_dropout = 0.05,
    bias = "none",
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    use_gradient_checkpointing = "unsloth",
    random_state = 42,
)


# In[8]:


from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./qwen-telecom-output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",

    logging_strategy="steps",
    logging_steps=200,

    num_train_epochs=3,
    save_strategy="epoch",

    eval_strategy="steps",
    eval_steps=200,

    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),

    report_to=["wandb"],
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    args=training_args,
)


# In[9]:


trainer.train()


# In[ ]:


from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch

FastLanguageModel.for_inference(model)

system_prompt = (
    "Bạn là chuyên gia Viễn thông cao cấp. "
    "Bạn sẽ giải đáp các câu hỏi từ người dùng một cách đầy đủ bằng Tiếng Việt."
)

text_streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

print("Gõ câu hỏi (gõ 'exit' để thoát)\n")

while True:
    user_question = input("User: ")
    if user_question.lower() in ["exit", "quit", "q"]:
        print("Thoát.")
        break

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=True
    ).to("cuda")

    print("Assistant:", end=" ", flush=True)

    with torch.no_grad():
        _ = model.generate(
            input_ids=inputs,
            streamer=text_streamer,
            max_new_tokens=512,
            use_cache=True,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
        )

    print("\n" + "-" * 60)


# In[ ]:


trainer.save_model("./qwen-telecom-final")

