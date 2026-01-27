from src.data.qna_loader import QNALoader
from src.data.mcq_loader import MCQLoader

q = QNALoader()
t = q.load()
print(t)

m = MCQLoader()
t = m.load()
print(t)
