"""
In-memory conversation store with sliding window.
"""

MAX_ROUNDS = 8  # 8 user + 8 assistant = 16 messages max

_conversations: dict[str, list[dict]] = {}


def get_history(session_id: str) -> list[dict]:
    history = _conversations.get(session_id, [])
    # Trim to sliding window
    if len(history) > MAX_ROUNDS * 2:
        history = history[-(MAX_ROUNDS * 2):]
        _conversations[session_id] = history
    return history


def save_round(session_id: str, user_msg: str, assistant_msg: str):
    history = _conversations.setdefault(session_id, [])
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})


def clear(session_id: str):
    _conversations.pop(session_id, None)
