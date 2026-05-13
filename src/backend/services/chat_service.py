from typing import AsyncGenerator, Optional, List
from llm_serve import TelcoLLM
from models.session import ChatMessageCreate
from services.session_service import add_message_to_session

async def process_chat_message(
    telco_llm: TelcoLLM,
    session_id: str, 
    message: str,
    file_ids: Optional[List[str]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    think: Optional[bool] = None,
    user_context: Optional[str] = None,
) -> str:
    # 1. Save user message
    user_msg = ChatMessageCreate(role="user", content=message, file_ids=file_ids)
    await add_message_to_session(session_id, user_msg)
    
    # 2. Call LangChain/OpenAI-compatible vLLM endpoint
    ai_response_content = await telco_llm.response(
        message=message,
        stream=False,
        file_ids=file_ids,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        think=think,
        user_context=user_context,
    )
    
    # 3. Save AI response
    ai_msg = ChatMessageCreate(role="ai", content=ai_response_content)
    await add_message_to_session(session_id, ai_msg)
    
    return ai_response_content

async def process_chat_message_stream(
    telco_llm: TelcoLLM,
    session_id: str, 
    message: str,
    file_ids: Optional[List[str]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    think: Optional[bool] = None,
    user_context: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    # 1. Save user message
    user_msg = ChatMessageCreate(role="user", content=message, file_ids=file_ids)
    await add_message_to_session(session_id, user_msg)
    
    # 2. Stream LangChain/OpenAI-compatible vLLM response
    response_stream = await telco_llm.response(
        message=message,
        stream=True,
        file_ids=file_ids,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        think=think,
        user_context=user_context,
    )

    chunks = []
    async for chunk in response_stream:
        chunks.append(chunk)
        yield chunk
        
    # 3. Save AI response after stream completes
    ai_response_content = "".join(chunks)
    ai_msg = ChatMessageCreate(role="ai", content=ai_response_content)
    await add_message_to_session(session_id, ai_msg)
