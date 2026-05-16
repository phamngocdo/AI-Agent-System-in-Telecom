import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import settings
from .constants import MARKDOWN_SEPARATORS


def build_markdown_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.FILE_RAG_CHUNK_SIZE,
        chunk_overlap=settings.FILE_RAG_CHUNK_OVERLAP,
        separators=MARKDOWN_SEPARATORS,
    )


def split_markdown(markdown: str, splitter: RecursiveCharacterTextSplitter) -> list[str]:
    normalized = re.sub(r"\n{3,}", "\n\n", markdown or "").strip()
    return [
        document.page_content.strip()
        for document in splitter.create_documents([normalized])
        if document.page_content.strip()
    ]
