from dataclasses import dataclass

from .conversation_memory import ConversationMemory
from .file_rag import DocumentRAGService
from .telco_llm import TelcoLLM


@dataclass
class LLMRuntime:
    telco_llm: TelcoLLM
    document_rag: DocumentRAGService
    conversation_memory: ConversationMemory

    @classmethod
    def create(cls) -> "LLMRuntime":
        return cls(
            telco_llm=TelcoLLM(),
            document_rag=DocumentRAGService(),
            conversation_memory=ConversationMemory(max_messages=5),
        )
