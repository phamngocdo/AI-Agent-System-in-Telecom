from typing import AsyncGenerator, Optional, Sequence

from fastapi import UploadFile

from llm_serve import FileProcessingCancelled, LLMRuntime
from llm_serve.thinking import strip_thinking
from models.session import ChatMessageCreate
from services.session_service import add_file_ids_to_session, add_message_to_session

FILE_ANALYSIS_STOPPED_MESSAGE = "Người dùng đã dừng việc phân tích tệp."


class ChatService:
    def __init__(self, runtime: LLMRuntime) -> None:
        self.runtime = runtime

    async def process_message(
        self,
        session_id: str,
        message: str,
        user_id: str,
        file_ids: Optional[list[str]] = None,
        uploaded_files: Optional[Sequence[UploadFile]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        think: Optional[bool] = None,
        user_context: Optional[str] = None,
    ) -> str:
        conversation_history = await self.runtime.conversation_memory.load(session_id)
        try:
            new_file_ids = await self._ingest_uploaded_files(
                session_id=session_id,
                user_id=user_id,
                uploaded_files=uploaded_files,
            )
        except FileProcessingCancelled:
            return FILE_ANALYSIS_STOPPED_MESSAGE

        effective_file_ids = await self._load_effective_file_ids(session_id, file_ids, new_file_ids)
        rag_context = await self._build_rag_context(
            session_id=session_id,
            user_id=user_id,
            message=message,
            file_ids=effective_file_ids,
            conversation_history=conversation_history,
        )

        ai_response_content = await self.runtime.telco_llm.response(
            message=message,
            stream=False,
            file_ids=effective_file_ids,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            think=think,
            user_context=user_context,
            conversation_history=conversation_history,
            rag_context=rag_context,
        )
        ai_response_content = strip_thinking(ai_response_content)

        await self._save_successful_exchange(
            session_id,
            message,
            ai_response_content,
            self._message_file_ids(file_ids, new_file_ids, effective_file_ids),
        )

        return ai_response_content

    async def process_message_stream(
        self,
        session_id: str,
        message: str,
        user_id: str,
        file_ids: Optional[list[str]] = None,
        uploaded_files: Optional[Sequence[UploadFile]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        think: Optional[bool] = None,
        user_context: Optional[str] = None,
    ) -> AsyncGenerator[str | dict, None]:
        conversation_history = await self.runtime.conversation_memory.load(session_id)
        new_file_ids: list[str] = []
        if uploaded_files:
            yield {"event": "status", "content": "Đang phân tích tệp..."}
            try:
                new_file_ids = await self._ingest_uploaded_files(
                    session_id=session_id,
                    user_id=user_id,
                    uploaded_files=uploaded_files,
                )
            except FileProcessingCancelled:
                yield {"event": "content", "content": FILE_ANALYSIS_STOPPED_MESSAGE}
                return
            yield {"event": "status", "content": "Đã phân tích tệp. Đang tìm thông tin liên quan..."}

        effective_file_ids = await self._load_effective_file_ids(session_id, file_ids, new_file_ids)
        rag_context = await self._build_rag_context(
            session_id=session_id,
            user_id=user_id,
            message=message,
            file_ids=effective_file_ids,
            conversation_history=conversation_history,
        )

        response_stream = await self.runtime.telco_llm.response(
            message=message,
            stream=True,
            file_ids=effective_file_ids,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            think=think,
            user_context=user_context,
            conversation_history=conversation_history,
            rag_context=rag_context,
        )

        chunks = []
        async for chunk in response_stream:
            chunks.append(chunk)
            yield chunk

        ai_response_content = strip_thinking("".join(chunks))
        await self._save_successful_exchange(
            session_id,
            message,
            ai_response_content,
            self._message_file_ids(file_ids, new_file_ids, effective_file_ids),
        )

    async def _save_successful_exchange(
        self,
        session_id: str,
        user_content: str,
        ai_content: str,
        file_ids: Optional[list[str]] = None,
    ) -> None:
        user_msg = ChatMessageCreate(role="user", content=user_content, file_ids=file_ids)
        await add_message_to_session(session_id, user_msg)

        ai_msg = ChatMessageCreate(role="ai", content=ai_content)
        await add_message_to_session(session_id, ai_msg)

    async def _ingest_uploaded_files(
        self,
        *,
        session_id: str,
        user_id: str,
        uploaded_files: Optional[Sequence[UploadFile]],
    ) -> list[str]:
        if not uploaded_files:
            return []

        uploaded_documents = await self.runtime.document_rag.ingest_uploads(
            session_id=session_id,
            user_id=user_id,
            files=uploaded_files,
        )
        new_file_ids = [document.file_id for document in uploaded_documents]
        await add_file_ids_to_session(session_id, user_id, new_file_ids)
        return new_file_ids

    async def _load_effective_file_ids(
        self,
        session_id: str,
        request_file_ids: Optional[Sequence[str]],
        new_file_ids: Optional[Sequence[str]],
    ) -> list[str]:
        session_file_ids = await self.runtime.conversation_memory.load_file_ids(session_id)
        session_file_id_set = set(session_file_ids)
        allowed_request_file_ids = [
            file_id for file_id in (request_file_ids or []) if file_id in session_file_id_set
        ]

        if request_file_ids is not None:
            return self._dedupe_file_ids([
                *allowed_request_file_ids,
                *(new_file_ids or []),
            ])

        return self._dedupe_file_ids([
            *(session_file_ids or []),
            *(new_file_ids or []),
        ])

    async def _build_rag_context(
        self,
        *,
        session_id: str,
        user_id: str,
        message: str,
        file_ids: Sequence[str],
        conversation_history: Sequence[tuple[str, str]],
    ) -> Optional[str]:
        if not file_ids:
            return None

        return await self.runtime.document_rag.build_context(
            session_id=session_id,
            user_id=user_id,
            telco_llm=self.runtime.telco_llm,
            question=message,
            file_ids=file_ids,
            conversation_history=conversation_history,
        )

    def _message_file_ids(
        self,
        request_file_ids: Optional[Sequence[str]],
        new_file_ids: Optional[Sequence[str]],
        effective_file_ids: Sequence[str],
    ) -> Optional[list[str]]:
        allowed = set(effective_file_ids)
        message_file_ids = self._dedupe_file_ids([
            *[file_id for file_id in (request_file_ids or []) if file_id in allowed],
            *(new_file_ids or []),
        ])
        return message_file_ids or None

    @staticmethod
    def _dedupe_file_ids(file_ids: Sequence[str]) -> list[str]:
        seen = set()
        result = []
        for file_id in file_ids:
            normalized = str(file_id).strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        return result
