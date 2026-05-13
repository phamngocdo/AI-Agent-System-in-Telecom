from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ChatMessageBase(BaseModel):
    role: str
    content: str
    file_ids: Optional[List[str]] = None

class ChatMessage(ChatMessageBase):
    id: str = Field(alias="_id")
    created_at: datetime
    
    class Config:
        populate_by_name = True

class ChatMessageCreate(ChatMessageBase):
    pass

class ChatSessionBase(BaseModel):
    title: str

class ChatSession(ChatSessionBase):
    id: str = Field(alias="_id")
    user_id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        populate_by_name = True

class ChatSessionCreate(ChatSessionBase):
    pass

class ChatSessionUpdate(BaseModel):
    title: str
