# utils/session_manager.py
import uuid
from datetime import datetime, timezone, timedelta
from threading import Lock

SESSION_TTL_MINUTES = 60  # Idle timeout for a session

class SessionManager:
    _sessions = {}
    _user_sessions = {}
    _lock = Lock()

    @staticmethod
    def _now():
        return datetime.now(timezone.utc)
    
    @classmethod
    def create_session(cls, user_id: str) -> str:
        with cls._lock:
            cls._cleanup_expired_sessions()

            session_id = str(uuid.uuid4())
            session_data = {
                "session_id": session_id,
                "user_id": user_id,
                "created_at": cls._now(),
                "last_accessed": cls._now(),
                "active_conversation_id": None
            }
            cls._sessions[session_id] = session_data
            cls._user_sessions.setdefault(user_id, []).append(session_id)
            return session_id

    @classmethod
    def get_session(cls, session_id: str) -> dict:
        with cls._lock:
            session = cls._sessions.get(session_id)
            if session:
                session["last_accessed"] = cls._now()
            return session

    @classmethod
    def get_user_sessions(cls, user_id: str) -> list:
        with cls._lock:
            session_ids = cls._user_sessions.get(user_id, [])
            return [cls._sessions[sid] for sid in session_ids if sid in cls._sessions]

    @classmethod
    def get_user_session(cls, user_id: str) -> str:
        with cls._lock:
            session_ids = cls._user_sessions.get(user_id, [])
            for sid in session_ids:
                session = cls._sessions.get(sid)
                if session:
                    return sid
            return None

    @classmethod
    def set_active_conversation(cls, session_id: str, conversation_id: str) -> bool:
        with cls._lock:
            session = cls._sessions.get(session_id)
            if session:
                session["active_conversation_id"] = conversation_id
                session["last_accessed"] = cls._now()
                return True
            return False

    @classmethod
    def get_active_conversation(cls, session_id: str) -> str:
        with cls._lock:
            session = cls._sessions.get(session_id)
            if session:
                return session.get("active_conversation_id")
            return None

    @classmethod
    def delete_session(cls, session_id: str) -> bool:
        with cls._lock:
            session = cls._sessions.pop(session_id, None)
            if session:
                user_id = session["user_id"]
                cls._user_sessions[user_id] = [
                    sid for sid in cls._user_sessions.get(user_id, []) if sid != session_id
                ]
                return True
            return False

    @classmethod
    def _cleanup_expired_sessions(cls):
        now = cls._now()
        expired = [
            sid for sid, session in cls._sessions.items()
            if now - session["last_accessed"] > timedelta(minutes=SESSION_TTL_MINUTES)
        ]
        for sid in expired:
            cls.delete_session(sid)
