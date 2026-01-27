from typing import List, Dict, Any
from src.data.data_loader import DatasetLoader

class QNALoader(DatasetLoader):
    def __init__(self, tokenizer, repo_id="phamngocdo/Vietnamese_TeleQnA_dataset"):
        super().__init__(tokenizer=tokenizer, repo_id=repo_id)
        self.dataset_name = "qna"

    def formatting_train_prompts_func(self, examples: Dict[str, Any]) -> Dict[str, List[Any]]:

        questions = examples["question"]
        answers   = examples["answer"]
        thinkings = examples["thinking"]

        texts = []

        system_prompt = (
            "Bạn là chuyên gia Viễn thông cao cấp. "
            "Hãy phân tích câu hỏi, suy luận logic trong thẻ <think>, "
            "sau đó đưa ra câu trả lời chính xác."
        )

        for q, ans, think in zip(questions, answers, thinkings):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
                {
                    "role": "assistant",
                    "content": f"<think>\n{think}\n</think>\n{ans}",
                },
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

    def formatting_test_prompts_func(self, examples: Dict[str, Any]) -> Dict[str, List[Any]]:

        questions  = examples["question"]

        texts = []

        system_prompt = (
            "Bạn là chuyên gia Viễn thông cao cấp. "
            "Hãy phân tích câu hỏi và đưa ra câu trả lời ngắn gọn, chính xác."
        )

        for q in questions:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
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
        }
