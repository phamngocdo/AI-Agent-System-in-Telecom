import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status

from database import get_database
from .constants import (
    MARKDOWN_CONTENT_TYPES,
    MAX_UPLOAD_FILES,
    MAX_UPLOAD_TOTAL_BYTES,
    PDF_CONTENT_TYPES,
)
from .exceptions import FileProcessingCancelled
from .hf_models import LocalCrossEncoderReranker, LocalEmbeddingModel
from .ocr import MarkerPDFProcessor
from .qdrant_store import QdrantChunkStore
from .retriever import DocumentRetriever
from .schemas import UploadedDocument
from .text_splitter import build_markdown_splitter, split_markdown


@dataclass(frozen=True)
class PreparedUpload:
    filename: str
    data: bytes
    file_type: str


class DocumentRAGService:
    def __init__(self) -> None:
        self.ocr = MarkerPDFProcessor()
        self.splitter = build_markdown_splitter()
        self.embeddings = LocalEmbeddingModel()
        self.chunk_store = QdrantChunkStore()
        self.retriever = DocumentRetriever(
            embeddings=self.embeddings,
            chunk_store=self.chunk_store,
            reranker=LocalCrossEncoderReranker(),
        )

    async def ingest_uploads(
        self,
        *,
        session_id: str,
        user_id: str,
        files: Sequence[UploadFile],
    ) -> list[UploadedDocument]:
        uploaded_documents: list[UploadedDocument] = []
        prepared_uploads = await self._read_and_validate_uploads(files)

        for upload in prepared_uploads:
            file_id = str(uuid4())
            filename = upload.filename

            await self._save_file_metadata(
                file_id=file_id,
                session_id=session_id,
                user_id=user_id,
                filename=filename,
                file_type=upload.file_type,
                status_value="processing",
            )

            try:
                document = await self._process_file(
                    data=upload.data,
                    filename=filename,
                    file_type=upload.file_type,
                    file_id=file_id,
                    session_id=session_id,
                    user_id=user_id,
                )
            except asyncio.CancelledError as exc:
                await self._save_file_metadata(
                    file_id=file_id,
                    session_id=session_id,
                    user_id=user_id,
                    filename=filename,
                    file_type=upload.file_type,
                    status_value="cancelled",
                )
                raise FileProcessingCancelled("Người dùng đã dừng việc phân tích tệp.") from exc
            except Exception:
                await self._save_file_metadata(
                    file_id=file_id,
                    session_id=session_id,
                    user_id=user_id,
                    filename=filename,
                    file_type=upload.file_type,
                    status_value="failed",
                )
                raise

            uploaded_documents.append(document)

        return uploaded_documents

    async def build_context(
        self,
        *,
        session_id: str,
        user_id: str,
        telco_llm,
        question: str,
        file_ids: Sequence[str],
        conversation_history: Optional[Sequence[tuple[str, str]]] = None,
    ) -> Optional[str]:
        return await self.retriever.build_context(
            session_id=session_id,
            user_id=user_id,
            telco_llm=telco_llm,
            question=question,
            file_ids=file_ids,
            conversation_history=conversation_history,
        )

    async def _process_file(
        self,
        *,
        data: bytes,
        filename: str,
        file_type: str,
        file_id: str,
        session_id: str,
        user_id: str,
    ) -> UploadedDocument:
        markdown = (
            self._decode_markdown(data)
            if file_type == "md"
            else await self.ocr.to_markdown(data, filename)
        )
        chunks = split_markdown(markdown, self.splitter)
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Không trích xuất được nội dung từ file {filename}.",
            )

        vectors = await self.embeddings.embed_documents(chunks)
        await self.chunk_store.upsert_chunks(
            vectors=vectors,
            chunks=chunks,
            file_id=file_id,
            session_id=session_id,
            user_id=user_id,
            filename=filename,
            file_type=file_type,
        )
        await self._save_file_metadata(
            file_id=file_id,
            session_id=session_id,
            user_id=user_id,
            filename=filename,
            file_type=file_type,
            status_value="completed",
            chunk_count=len(chunks),
        )

        return UploadedDocument(
            file_id=file_id,
            filename=filename,
            file_type=file_type,
            chunk_count=len(chunks),
        )

    async def _read_and_validate_uploads(
        self,
        files: Sequence[UploadFile],
    ) -> list[PreparedUpload]:
        if not files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cần upload ít nhất một file PDF hoặc Markdown.",
            )
        if len(files) > MAX_UPLOAD_FILES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Chỉ được upload tối đa {MAX_UPLOAD_FILES} file.",
            )

        prepared: list[PreparedUpload] = []
        total_size = 0
        for upload in files:
            filename = Path(upload.filename or "").name
            suffix = Path(filename).suffix.lower()
            content_type = (upload.content_type or "").split(";")[0].strip().lower()

            if suffix not in {".pdf", ".md"}:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Chỉ hỗ trợ file PDF hoặc Markdown: {filename or 'không rõ tên'}.",
                )

            data = await upload.read(MAX_UPLOAD_TOTAL_BYTES + 1)
            if not data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File không được rỗng: {filename}.",
                )

            total_size += len(data)
            if total_size > MAX_UPLOAD_TOTAL_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Tổng dung lượng file tối đa 3MB.",
                )

            if suffix == ".pdf":
                self._validate_pdf(filename, content_type, data)
                prepared.append(PreparedUpload(filename=filename, data=data, file_type="pdf"))
                continue

            self._validate_markdown(filename, content_type, data)
            prepared.append(PreparedUpload(filename=filename, data=data, file_type="md"))

        return prepared

    @staticmethod
    def _validate_pdf(filename: str, content_type: str, data: bytes) -> None:
        if content_type and content_type not in PDF_CONTENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File PDF không hợp lệ: {filename}.",
            )
        if not data.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File upload không phải PDF hợp lệ: {filename}.",
            )

    @staticmethod
    def _validate_markdown(filename: str, content_type: str, data: bytes) -> None:
        if content_type and content_type not in MARKDOWN_CONTENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File Markdown không hợp lệ: {filename}.",
            )
        try:
            data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File Markdown cần dùng mã hóa UTF-8: {filename}.",
            ) from exc

    @staticmethod
    def _decode_markdown(data: bytes) -> str:
        return data.decode("utf-8-sig").strip()

    @staticmethod
    async def _save_file_metadata(
        *,
        file_id: str,
        session_id: str,
        user_id: str,
        filename: str,
        file_type: str,
        status_value: str,
        chunk_count: int = 0,
    ) -> None:
        db = get_database()
        now = datetime.utcnow()
        await db["files"].update_one(
            {"file_id": file_id},
            {
                "$set": {
                    "file_id": file_id,
                    "session_id": session_id,
                    "user_id": user_id,
                    "filename": filename,
                    "file_type": file_type,
                    "status": status_value,
                    "chunk_count": chunk_count,
                    "updated_at": now,
                },
                "$setOnInsert": {
                    "created_at": now,
                }
            },
            upsert=True,
        )
