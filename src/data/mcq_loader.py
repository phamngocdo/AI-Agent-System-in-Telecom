from typing import List, Dict, Any, Tuple
from src.data.data_loader import DatasetLoader

class MCQLoader(DatasetLoader):
    def __init__(self, tokenizer, repo_id="phamngocdo/Vietnamese_TeleMCQ_dataset"):
        super().__init__(tokenizer=tokenizer, repo_id=repo_id)
        self.dataset_name = "mcq"

    def formatting_train_prompts_func(examples: Dict[str, Any]) -> Dict[str, List[str]]:
        questions     = examples["question"]
        choices_dicts = examples["choices"]
        answers       = examples["answer"]
        explanations  = examples["explanation"]
        thinkings     = examples["thinking"]

        texts = []

        system_prompt = (
            "Bạn là chuyên gia Viễn thông cao cấp. "
            "Hãy phân tích câu hỏi trắc nghiệm, thực hiện suy luận logic trong thẻ <think> "
            "trước khi đưa ra đáp án cuối cùng."
        )

        for q, choices, ans_idx, expl, think in zip(
            questions, choices_dicts, answers, explanations, thinkings
        ):
            formatted_choices = ""
            for k in sorted(choices.keys(), key=int):
                formatted_choices += f"{k}. {choices[k]}\n"

            user_content = f"{q}\n\nLựa chọn:\n{formatted_choices}"

            ans_key = str(ans_idx)
            correct_choice_text = choices.get(ans_key, "Không xác định")
            response_label = f"Phương án {ans_key}"

            assistant_content = (
                f"<think>\n{think}\n</think>\n"
                f"{expl}\n"
                f"Đáp án đúng là: {response_label}. {correct_choice_text}"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

        return {
            "text": texts,
            "category": examples["category"],
        }

    def formatting_test_prompts_func(examples: Dict[str, Any]) -> Dict[str, List[str]]:
        questions     = examples["question"]
        choices_dicts = examples["choices"]

        texts = []

        system_prompt = (
            "Bạn là chuyên gia Viễn thông cao cấp. "
            "Hãy phân tích câu hỏi trắc nghiệm và chọn phương án đúng nhất."
        )

        for q, choices in zip(questions, choices_dicts):
            formatted_choices = ""
            for k in sorted(choices.keys(), key=int):
                formatted_choices += f"{k}. {choices[k]}\n"

            user_content = f"{q}\n\nLựa chọn 1 trong các đáp án sau:\n{formatted_choices}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True, 
            )
            texts.append(text)

        return {
            "text": texts,
            "category": examples["category"],
            "answer": examples["answer"],
            "explanation": examples["explanation"],
        }
