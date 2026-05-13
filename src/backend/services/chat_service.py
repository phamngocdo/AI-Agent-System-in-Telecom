from typing import AsyncGenerator, Optional, List
from llm_serve import ConversationMemory, TelcoLLM
from llm_serve.thinking import strip_thinking
from models.session import ChatMessageCreate
from services.session_service import add_message_to_session

conversation_memory = ConversationMemory(max_messages=5)

async def save_successful_exchange(
    session_id: str,
    user_content: str,
    ai_content: str,
    file_ids: Optional[List[str]] = None,
) -> None:
    user_msg = ChatMessageCreate(role="user", content=user_content, file_ids=file_ids)
    await add_message_to_session(session_id, user_msg)

    ai_msg = ChatMessageCreate(role="ai", content=ai_content)
    await add_message_to_session(session_id, ai_msg)

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
    conversation_history = await conversation_memory.load(session_id)
    
    ai_response_content = await telco_llm.response(
        message=message,
        stream=False,
        file_ids=file_ids,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        think=think,
        user_context=user_context,
        conversation_history=conversation_history,
    )
    ai_response_content = strip_thinking(ai_response_content)
    
    await save_successful_exchange(session_id, message, ai_response_content, file_ids)
    
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
    conversation_history = await conversation_memory.load(session_id)
    
    response_stream = await telco_llm.response(
        message=message,
        stream=True,
        file_ids=file_ids,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        think=think,
        user_context=user_context,
        conversation_history=conversation_history,
    )

    chunks = []
    async for chunk in response_stream:
        chunks.append(chunk)
        yield chunk
        
    ai_response_content = strip_thinking("".join(chunks))
    await save_successful_exchange(session_id, message, ai_response_content, file_ids)
