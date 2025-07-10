import tiktoken
from typing import List, Dict, Any, Optional
from datetime import datetime
from utils.ports.conversation_repository import ConversationRepository


class ConversationManager:
    def __init__(self, repository: ConversationRepository, max_tokens: int = 2000, model_name: str = "gpt-3.5-turbo"):
        self.repo = repository
        self.max_tokens = max_tokens
        self.encoder = tiktoken.encoding_for_model(model_name)

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def estimate_message_tokens(self, message: Dict[str, Any]) -> int:
        content = message.get("content", "")
        if isinstance(content, dict):
            # Prefer the 'response' field if present, else fallback to string conversion
            content = content.get("response", str(content))
        content_tokens = self.count_tokens(content)
        return content_tokens + 6

    def create_conversation(self, session_id: str, title: Optional[str] = None, model_info: Optional[Dict] = None) -> str:
        return self.repo.create_conversation(session_id, title, model_info)

    def add_message(self, conversation_id: str, role: str, content: str, intent: str, model_name: str):
        self.repo.add_message(conversation_id, role, content, intent, model_name)

    def get_conversations(self, user_id: str):
        return self.repo.get_conversations(user_id)

    def get_messages(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        return self.repo.get_messages(conversation_id)

    def filter_context_by_intent_with_token_limit(
        self,
        conversation_id: str,
        current_intent: str,
        max_tokens: Optional[int] = None,
        max_messages: int = 15
    ) -> List[Dict[str, Any]]:
        conversation = self.repo.get_conversation(conversation_id)
        if not conversation:
            return []

        max_tokens = max_tokens or self.max_tokens
        all_messages = self.repo.get_messages(conversation_id)

        system_messages = [msg for msg in all_messages if msg.get("role") == "system"]
        filtered_messages = []
        token_count = 0

        for msg in system_messages:
            msg_tokens = self.estimate_message_tokens(msg)
            if token_count + msg_tokens <= max_tokens:
                filtered_messages.append(msg)
                token_count += msg_tokens

        recent_messages = []
        user_assistant_pairs = 0

        for msg in reversed(all_messages):
            if msg.get("role") != "system":
                recent_messages.insert(0, msg)
                if msg.get("role") == "user":
                    user_assistant_pairs += 1
                    if user_assistant_pairs >= 5:
                        break

        intent_related = [
            msg for msg in all_messages
            if msg.get("role") != "system" and msg not in recent_messages and msg.get("intent") == current_intent
        ]

        candidate_messages = intent_related + recent_messages
        seen = set()
        unique_candidates = []
        for msg in candidate_messages:
            msg_id = msg.get("id", f"{msg.get('role')}_{msg.get('timestamp')}")
            if msg_id not in seen:
                seen.add(msg_id)
                unique_candidates.append(msg)

        selected_messages = []
        current_tokens = token_count
        for msg in reversed(unique_candidates):
            msg_tokens = self.estimate_message_tokens(msg)
            if current_tokens + msg_tokens <= max_tokens:
                selected_messages.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break

        final_messages = system_messages + self._ensure_conversation_flow(selected_messages)
        return final_messages[:max_messages]

    def _ensure_conversation_flow(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not messages:
            return messages

        corrected = []
        last_role = None
        for msg in messages:
            current_role = msg.get("role")
            if current_role != last_role or current_role == "system":
                corrected.append(msg)
                last_role = current_role
        return corrected

    def create_conversation_summary(self, conversation_id: str, intent: str = None) -> str:
        conversation = self.repo.get_conversation(conversation_id)
        if not conversation:
            return ""

        messages = conversation.get("messages", [])
        if intent:
            messages = [msg for msg in messages if msg.get("intent") == intent]

        if len(messages) < 6:
            return ""

        user_topics = [msg.get("content", "")[:100] for msg in messages if msg.get("role") == "user"]
        assistant_responses = [msg.get("content", "")[:100] for msg in messages if msg.get("role") == "assistant"]

        topics = ", ".join(user_topics[:3])
        summary = f"Previous discussion covered: {topics}. Assistant provided help with these topics."
        return summary

    def get_smart_context(self, conversation_id: str, current_intent: str, include_summary: bool = True) -> List[Dict[str, Any]]:
        conversation = self.repo.get_conversation(conversation_id)
        if not conversation:
            return []

        message_count = conversation.num_messages
        if message_count > 30 and include_summary:
            summary = self.create_conversation_summary(conversation_id, current_intent)
            recent_context = self.filter_context_by_intent_with_token_limit(
                conversation_id, current_intent, max_tokens=1500, max_messages=10
            )

            if summary:
                summary_msg = {
                    "role": "system",
                    "content": f"Previous conversation summary: {summary}",
                    "timestamp": datetime.now().isoformat()
                }
                return [summary_msg] + recent_context

        return self.filter_context_by_intent_with_token_limit(conversation_id, current_intent)

    def filter_context_by_intent(self, conversation_id: str, current_intent: str) -> List[Dict[str, Any]]:
        return self.filter_context_by_intent_with_token_limit(conversation_id, current_intent, max_tokens=self.max_tokens)
