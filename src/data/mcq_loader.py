from typing import List, Dict, Any
from src.data.data_loader import DatasetLoader

class MCQLoader(DatasetLoader):
    def __init__(self, tokenizer, repo_id="phamngocdo/Vietnamese_TeleMCQ_dataset"):
        super().__init__(tokenizer=tokenizer, repo_id=repo_id)
        self.tokenizer = tokenizer
        self.dataset_name = "mcq"

    @staticmethod
    def _normalize_choices(choices: Dict[str, Any]) -> Dict[str, str]:
        """
        Remove None / empty choices.
        Keep original keys (do NOT reindex).
        """
        return {
            str(k): str(v).strip()
            for k, v in choices.items()
            if v is not None and str(v).strip() != ""
        }

    @staticmethod
    def _format_choices(choices: Dict[str, str]) -> str:
        """
        Format choices for prompt.
        """
        lines = []
        for k in sorted(choices.keys(), key=int):
            lines.append(f"{k}. {choices[k]}")
        return "\n".join(lines)

    def formatting_train_prompts_func(
        self, examples: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        questions     = examples["question"]
        choices_dicts = examples["choices"]
        answers       = examples["answer"]
        explanations  = examples["explanation"]
        thinkings     = examples["thinking"]
        categories    = examples["category"]

        texts = []

        system_prompt = (
            "Bạn là chuyên gia Viễn thông cao cấp. "
            "Hãy phân tích câu hỏi trắc nghiệm, thực hiện suy luận logic trong thẻ <think> "
            "trước khi đưa ra đáp án cuối cùng."
        )

        for q, raw_choices, ans_idx, expl, think in zip(
            questions, choices_dicts, answers, explanations, thinkings
        ):
            choices = self._normalize_choices(raw_choices)

            formatted_choices = self._format_choices(choices)
            user_content = f"{q}\n\nLựa chọn:\n{formatted_choices}"

            ans_key = str(ans_idx)
            correct_choice_text = choices.get(ans_key)

            if correct_choice_text is None:
                correct_choice_text = "Không xác định"

            assistant_content = (
                f"<think>\n{think}\n</think>\n"
                f"{expl}\n"
                f"Đáp án đúng là: Phương án {ans_key}. {correct_choice_text}"
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
            "category": categories,
        }

    def formatting_test_prompts_func(
        self, examples: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        questions     = examples["question"]
        choices_dicts = examples["choices"]
        categories    = examples["category"]

        texts = []

        system_prompt = (
            "Bạn là chuyên gia Viễn thông cao cấp. "
            "Hãy phân tích câu hỏi trắc nghiệm và chọn phương án đúng nhất."
        )

        for q, raw_choices in zip(questions, choices_dicts):
            choices = self._normalize_choices(raw_choices)

            formatted_choices = self._format_choices(choices)
            user_content = (
                f"{q}\n\n"
                f"Lựa chọn 1 trong các đáp án sau:\n{formatted_choices}"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)

        return {
            "text": texts,
            "category": categories,
            "answer": examples["answer"],
            "explanation": examples["explanation"],
        }
