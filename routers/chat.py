import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from config import LLM_API_KEY
from services.llm import build_messages, chat, chat_stream
from services.rag import retrieve
from store.memory import get_history, save_round, clear

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"
    use_rag: bool = True


@router.post("/chat/stream")
async def stream_endpoint(req: ChatRequest):
    """
    SSE streaming chat.
    Interview point: "SSE is server-push only, lighter than WebSocket,
    ideal for LLM token-by-token generation."
    """
    if not LLM_API_KEY:
        raise HTTPException(500, "LLM_API_KEY not configured")

    history = get_history(req.session_id)
    context = retrieve(req.query) if req.use_rag else ""
    messages = build_messages(history, req.query, context)

    async def generate():
        full = ""
        try:
            async for chunk in chat_stream(messages):
                full += chunk
                yield {"data": json.dumps({"content": chunk}, ensure_ascii=False)}

            save_round(req.session_id, req.query, full)
            yield {"data": "[DONE]"}
        except Exception as e:
            yield {"data": json.dumps({"error": str(e)}, ensure_ascii=False)}

    return EventSourceResponse(generate())


@router.post("/chat")
async def normal_endpoint(req: ChatRequest):
    """Non-streaming chat for quick testing."""
    if not LLM_API_KEY:
        raise HTTPException(500, "LLM_API_KEY not configured")

    history = get_history(req.session_id)
    context = retrieve(req.query) if req.use_rag else ""
    messages = build_messages(history, req.query, context)
    content = await chat(messages)
    save_round(req.session_id, req.query, content)
    return {"content": content, "session_id": req.session_id}


@router.delete("/chat/history/{session_id}")
async def clear_endpoint(session_id: str):
    clear(session_id)
    return {"message": f"History cleared for {session_id}"}
