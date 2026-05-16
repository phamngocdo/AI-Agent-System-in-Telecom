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
    file_ids: List[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    
    class Config:
        populate_by_name = True

class ChatSessionCreate(ChatSessionBase):
    pass

class ChatSessionUpdate(BaseModel):
    title: str

class ChatFile(BaseModel):
    file_id: str
    filename: str
    file_type: Optional[str] = None
    status: str
    chunk_count: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
