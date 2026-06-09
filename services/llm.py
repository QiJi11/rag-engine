"""
LLM client wrapper — OpenAI-compatible, works with GPT / DeepSeek / Qwen.
"""

from openai import AsyncOpenAI
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

SYSTEM_PROMPT = (
    "你是一个知识库问答助手。"
    "请基于提供的上下文信息回答用户问题。"
    "如果上下文中没有相关信息，请如实告知，不要编造答案。"
    "回答简洁专业。"
)


def build_messages(history: list[dict], query: str, context: str = "") -> list[dict]:
    """Assemble the full message list for the LLM call."""
    system = SYSTEM_PROMPT
    if context:
        system += f"\n\n以下是检索到的相关知识：\n{context}"

    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": query})
    return messages


async def chat(messages: list[dict]) -> str:
    """Non-streaming call, returns full response."""
    resp = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=2048,
    )
    return resp.choices[0].message.content


async def chat_stream(messages: list[dict]):
    """Streaming call, yields content chunks."""
    stream = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=2048,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
