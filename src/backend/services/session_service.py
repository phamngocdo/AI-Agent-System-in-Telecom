from typing import List, Optional
from datetime import datetime
from random import randint

from bson import ObjectId
from database import get_database
from models.session import ChatSessionCreate, ChatMessageCreate, ChatSessionUpdate

async def _get_unique_session_title(
    db,
    user_id: str,
    title: str,
    exclude_session_id: Optional[str] = None,
) -> str:
    base_title = title.strip() if title and title.strip() else "Hội thoại"
    candidate = base_title

    query = {"user_id": user_id, "title": candidate}
    if exclude_session_id:
        query["_id"] = {"$ne": ObjectId(exclude_session_id)}

    for _ in range(100):
        query["title"] = candidate
        existing = await db["sessions"].find_one(query)
        if existing is None:
            return candidate
        candidate = f"{base_title} {randint(1000, 9999)}"

    return f"{base_title} {datetime.utcnow().strftime('%H%M%S%f')[-8:]}"

async def create_session(user_id: str, session: ChatSessionCreate) -> dict:
    db = get_database()
    now = datetime.utcnow()
    session_dict = session.model_dump()
    session_dict["title"] = await _get_unique_session_title(db, user_id, session_dict.get("title", ""))
    session_dict.update({
        "user_id": user_id,
        "created_at": now,
        "updated_at": now
    })
    result = await db["sessions"].insert_one(session_dict)
    session_dict["_id"] = str(result.inserted_id)
    return session_dict

async def get_user_sessions(user_id: str) -> List[dict]:
    db = get_database()
    cursor = db["sessions"].find({"user_id": user_id}).sort("updated_at", -1)
    sessions = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        sessions.append(doc)
    return sessions

async def get_session(session_id: str, user_id: str) -> Optional[dict]:
    db = get_database()
    try:
        session = await db["sessions"].find_one({"_id": ObjectId(session_id), "user_id": user_id})
        if session:
            session["_id"] = str(session["_id"])
        return session
    except Exception:
        return None

async def update_session(
    session_id: str,
    user_id: str,
    session_update: ChatSessionUpdate,
) -> Optional[dict]:
    db = get_database()
    try:
        object_id = ObjectId(session_id)
        existing = await db["sessions"].find_one({"_id": object_id, "user_id": user_id})
        if existing is None:
            return None

        now = datetime.utcnow()
        title = await _get_unique_session_title(
            db,
            user_id,
            session_update.title,
            exclude_session_id=session_id,
        )
        await db["sessions"].update_one(
            {"_id": object_id, "user_id": user_id},
            {"$set": {"title": title, "updated_at": now}},
        )
        return await get_session(session_id, user_id)
    except Exception:
        return None

async def delete_session(session_id: str, user_id: str) -> bool:
    db = get_database()
    try:
        object_id = ObjectId(session_id)
        result = await db["sessions"].delete_one({"_id": object_id, "user_id": user_id})
        if result.deleted_count == 0:
            return False

        await db["messages"].delete_many({"session_id": session_id})
        return True
    except Exception:
        return False

async def add_message_to_session(session_id: str, message: ChatMessageCreate) -> dict:
    db = get_database()
    now = datetime.utcnow()
    message_dict = message.model_dump()
    message_dict.update({
        "session_id": session_id,
        "created_at": now
    })
    
    result = await db["messages"].insert_one(message_dict)
    message_dict["_id"] = str(result.inserted_id)
    
    # Update session's updated_at
    await db["sessions"].update_one(
        {"_id": ObjectId(session_id)},
        {"$set": {"updated_at": now}}
    )
    
    return message_dict

async def get_session_messages(session_id: str) -> List[dict]:
    db = get_database()
    cursor = db["messages"].find({"session_id": session_id}).sort("created_at", 1)
    messages = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        messages.append(doc)
    return messages
