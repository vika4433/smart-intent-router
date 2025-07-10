from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ConversationInfo:
    conversation_id: str
    user_id: str
    title: str
    created_at: datetime
    last_modified: datetime
    model_info: Optional[Dict]
    num_messages: int


class ConversationRepository(ABC):
    @abstractmethod
    def create_conversation(self, user_id: str, title: Optional[str], model_info: Optional[Dict]) -> str:
        """Create a new conversation for a user and return its ID."""
        pass

    @abstractmethod
    def get_conversation(self, conversation_id: str) -> Optional[ConversationInfo]:
        """Return conversation metadata (title, created, last_modified, etc.)"""
        pass

    @abstractmethod
    def delete_conversation(self, conversation_id: str) -> bool:
        pass

    @abstractmethod
    def get_conversations(self, user_id: str) -> List[ConversationInfo]:
        """Return all conversations for a user (metadata only, not messages)"""
        pass

    @abstractmethod
    def add_message(self, conversation_id: str, role: str, content: str, intent: str, model_name: str) -> bool:
        """
        Add a message to a conversation. Message must follow OpenAI format:
        {"role": "user|assistant|system", "content": "...", ...}
        Return True if message was added, False otherwise (e.g., if conversation is full).
        """
        pass

    @abstractmethod
    def get_messages(self, conversation_id: str, intent: Optional[str] = None) -> List[Dict]:
        """
        Return all messages for a conversation, each in OpenAI format:
        {"role": "user|assistant|system", "content": "...", ...}
        """
        pass

    @abstractmethod
    def clear_messages(self, conversation_id: str) -> None:
        pass

