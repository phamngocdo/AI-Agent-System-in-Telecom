from typing import Optional

from database import get_database


class ConversationMemory:
    def __init__(
        self,
        max_messages: int = 5,
        max_message_chars: Optional[int] = 3000,
    ) -> None:
        self.max_messages = max_messages
        self.max_message_chars = max_message_chars

    async def load(self, session_id: str) -> list[tuple[str, str]]:
        db = get_database()
        cursor = (
            db["messages"]
            .find({"session_id": session_id})
            .sort("created_at", -1)
            .limit(self.max_messages)
        )

        messages = []
        async for doc in cursor:
            messages.append(doc)

        messages.reverse()
        return [turn for doc in messages if (turn := self._to_chat_turn(doc))]

    def _to_chat_turn(self, message: dict) -> Optional[tuple[str, str]]:
        role = self._normalize_role(message.get("role"))
        content = self._normalize_content(message.get("content"))

        if not role or not content:
            return None

        return role, content

    @staticmethod
    def _normalize_role(role: Optional[str]) -> Optional[str]:
        role_map = {
            "user": "human",
            "human": "human",
            "ai": "ai",
            "assistant": "ai",
        }
        return role_map.get((role or "").strip().lower())

    def _normalize_content(self, content: Optional[str]) -> Optional[str]:
        if not content:
            return None

        normalized = str(content).strip()
        if not normalized:
            return None

        if self.max_message_chars is not None:
            return normalized[:self.max_message_chars]

        return normalized
