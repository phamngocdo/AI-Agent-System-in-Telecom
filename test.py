from src.training.trainer import get_model_and_tokenizer
from src.data.qna_loader import QNALoader

def run():
    _, tokenizer = get_model_and_tokenizer()

    qna_loader = QNALoader(tokenizer=tokenizer)

    datasets = qna_loader.load(splits=("train", "test"))

if __name__ == "__main__":
    run()