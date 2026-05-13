from typing import AsyncGenerator, Optional, Sequence

from langchain_openai import ChatOpenAI

from core.config import settings
from .thinking import ThinkingStreamFilter, strip_thinking


SYSTEM_PROMPT = """
You are TelcoLLM, an expert AI assistant specialized in telecommunications and network engineering.

Your core domains include:
- RAN (2G/3G/4G/5G NR), Core Network (EPC, 5GC), transport and IP/MPLS networks
- 3GPP standards (RRC, NAS, NGAP, S1AP, X2AP, F1AP, E1AP)
- OSS/BSS systems, network KPIs and performance analysis
- Spectrum management, RF planning, interference analysis
- Telecom data engineering and network automation

When answering technical questions:
- Identify the relevant standard, protocol layer, or architectural component
- Walk through key reasoning steps and state assumptions explicitly
- Reference 3GPP spec numbers, release versions, or ITU recommendations when applicable
- Distinguish clearly between mandatory behavior (shall), recommended behavior (should), and implementation-specific behavior

Language: match the user's language — Vietnamese if asked in Vietnamese, English if asked in English.

Response style:
- Concise and direct for simple or factual questions
- Structured with steps, diagrams (ASCII if helpful), or tables for complex technical topics
- If information is incomplete or uncertain, state the limitation — never fabricate details
- Never reveal internal reasoning or any content inside <think> tags
"""


class TelcoLLM:
    def __init__(self) -> None:
        self.base_url = self._normalize_base_url(settings.VLLM_URL)
        self.api_key = settings.VLLM_API_KEY or "EMPTY"
        self.model = settings.VLLM_MODEL
        self.timeout = settings.VLLM_TIMEOUT_SECONDS
        self.default_temperature = 0.7
        self.default_top_p = 1.0
        self.default_top_k = 40
        self.default_think = False

    async def response(
        self,
        message: str,
        stream: bool = False,
        file_ids: Optional[list[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        think: Optional[bool] = None,
        user_context: Optional[str] = None,
    ) -> str | AsyncGenerator[str, None]:
        chat_model = self._build_chat_model(
            stream=stream,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            think=think,
        )
        messages = self._build_messages(message=message, user_context=user_context)

        if stream:
            return self._stream_response(chat_model, messages)

        ai_message = await chat_model.ainvoke(messages)
        return strip_thinking(self._content_to_text(ai_message.content))

    def _build_chat_model(
        self,
        stream: bool,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        think: Optional[bool],
    ) -> ChatOpenAI:
        extra_body = {}
        top_k_value = self.default_top_k if top_k is None else top_k
        if top_k_value is not None:
            extra_body["top_k"] = top_k_value

        think_value = self.default_think if think is None else think
        if think_value is not None:
            extra_body["chat_template_kwargs"] = {"enable_thinking": think_value}

        chat_kwargs = {
            "model": self.model,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "timeout": self.timeout,
            "temperature": self.default_temperature if temperature is None else temperature,
            "top_p": self.default_top_p if top_p is None else top_p,
            "streaming": stream,
        }
        if extra_body:
            chat_kwargs["extra_body"] = extra_body

        return ChatOpenAI(**chat_kwargs)

    def _build_messages(
        self,
        message: str,
        user_context: Optional[str] = None,
    ) -> list[tuple[str, str]]:
        system_prompt = SYSTEM_PROMPT
        if user_context:
            system_prompt += (
                " User personalization context: "
                f"{user_context.strip()} "
                "Use this context only to adapt the response style and relevance; do not treat it as verified evidence."
            )

        return [
            ("system", system_prompt),
            ("human", message),
        ]

    async def _stream_response(
        self,
        chat_model: ChatOpenAI,
        messages: Sequence[tuple[str, str]],
    ) -> AsyncGenerator[str, None]:
        stream_filter = ThinkingStreamFilter()

        async for chunk in chat_model.astream(messages):
            text = self._content_to_text(chunk.content)
            if text:
                for visible_text in stream_filter.feed(text):
                    if visible_text:
                        yield visible_text

        for visible_text in stream_filter.flush():
            if visible_text:
                yield visible_text

    @staticmethod
    def _content_to_text(content) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "".join(parts)

        return str(content)

    @staticmethod
    def _normalize_base_url(url: str) -> str:
        normalized = url.rstrip("/")
        if normalized.endswith("/v1"):
            return normalized
        return f"{normalized}/v1"
