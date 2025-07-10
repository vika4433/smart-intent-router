from utils.ports.conversation_repository import ConversationRepository, ConversationInfo
from typing import Optional, List, Dict, Any
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
from dataclasses import dataclass, asdict

MESSAGE_LIMIT = 50  # Example limit, adjust as needed

@dataclass
class StoredMessage:
    role: str
    content: str
    intent: Optional[str] = None
    model: Optional[str] = None
    timestamp: Optional[Any] = None
    name: Optional[str] = None
    function_call: Optional[Dict] = None

    def to_openai(self) -> Dict:
        # Only include OpenAI-compatible fields
        msg = {"role": self.role, "content": self.content}
        if self.name:
            msg["name"] = self.name
        if self.function_call:
            msg["function_call"] = self.function_call
        return msg

class MongoConversationRepository(ConversationRepository):
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017", db_name: str = "smart_router"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.conversations = self.db["conversations"]

    def create_conversation(self, user_id: str, title: Optional[str], model_info: Optional[Dict]) -> str:
        conversation = {
            "user_id": user_id,
            "title": title or "Untitled",
            "created_at": datetime.now(),
            "last_modified": datetime.now(),
            "model_info": model_info or {},
            "messages": [],
        }
        result = self.conversations.insert_one(conversation)
        return str(result.inserted_id)

    def get_conversation(self, conversation_id: str) -> Optional[ConversationInfo]:
        convo = self.conversations.find_one({"_id": ObjectId(conversation_id)})
        if not convo:
            return None
        return ConversationInfo(
            conversation_id=str(convo["_id"]),
            user_id=convo.get("user_id"),
            title=convo.get("title"),
            created_at=convo.get("created_at"),
            last_modified=convo.get("last_modified"),
            model_info=convo.get("model_info", {}),
            num_messages=len(convo.get("messages", [])),
        )

    def delete_conversation(self, conversation_id: str) -> bool:
        result = self.conversations.delete_one({"_id": ObjectId(conversation_id)})
        return result.deleted_count == 1

    def get_conversations(self, user_id: str) -> List[ConversationInfo]:
        convos = self.conversations.find({"user_id": user_id})
        return [
            ConversationInfo(
                conversation_id=str(c["_id"]),
                user_id=c.get("user_id"),
                title=c.get("title"),
                created_at=c.get("created_at"),
                last_modified=c.get("last_modified"),
                model_info=c.get("model_info", {}),
                num_messages=len(c.get("messages", [])),
            )
            for c in convos
        ]

    def add_message(self, conversation_id: str, role: str, content: str, intent: str, model_name: str) -> bool:
        if role not in {"user", "assistant", "system"}:
            return False
        message = StoredMessage(
            role=role,
            content=content,
            intent=intent,
            model=model_name,
            timestamp=datetime.now()
        )
        convo = self.conversations.find_one({"_id": ObjectId(conversation_id)})
        if not convo:
            return False
        if len(convo.get("messages", [])) >= MESSAGE_LIMIT:
            return False
        result = self.conversations.update_one(
            {"_id": ObjectId(conversation_id)},
            {"$push": {"messages": asdict(message)}, "$set": {"last_modified": datetime.now()}}
        )
        return result.modified_count == 1

    def get_messages(self, conversation_id: str, intent: Optional[str] = None) -> List[Dict]:
        convo = self.conversations.find_one({"_id": ObjectId(conversation_id)})
        if not convo:
            return []
        messages = convo.get("messages", [])
        # Return the full message dicts (with all metadata)
        if intent:
            messages = [m for m in messages if m.get("intent") == intent]
        return messages

    # Optionally, if you need OpenAI format for UI or API, add:
    def get_openai_messages(self, conversation_id: str, intent: Optional[str] = None) -> List[Dict]:
        messages = self.get_messages(conversation_id, intent)
        return [StoredMessage(**m).to_openai() for m in messages]

    def clear_messages(self, conversation_id: str) -> None:
        self.conversations.update_one(
            {"_id": ObjectId(conversation_id)},
            {"$set": {"messages": [], "last_modified": datetime.now()}}
        )
