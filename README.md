# RAG Engine

基于 Python + FastAPI + ChromaDB 的 RAG 知识库问答后端 Demo，实现文档上传、分块、向量检索、CrossEncoder rerank、Prompt 注入和 SSE 流式回答链路。

## 技术栈

- Python 3.10+
- FastAPI
- ChromaDB
- SentenceTransformers
- CrossEncoder Rerank
- OpenAI-compatible Chat API
- SSE

## 项目结构

```text
rag-engine/
├── main.py
├── config.py
├── routers/
│   ├── chat.py
│   └── upload.py
├── services/
│   ├── chunker.py
│   ├── llm.py
│   └── rag.py
├── store/
│   ├── memory.py
│   └── vector_store.py
└── static/
    └── index.html
```

## 核心实现

- `POST /api/v1/upload`：支持 TXT / PDF 文档解析，按 512 字符分块与 50 字符 overlap 处理文本，并写入 ChromaDB 持久化集合。
- 向量检索：基于 `all-MiniLM-L6-v2` embedding function 管理 ChromaDB collection，查询时先粗召回候选片段。
- Rerank：使用 `cross-encoder/ms-marco-MiniLM-L-6-v2` 对候选片段重新排序，选出 Top-3 片段注入 System Prompt。
- `POST /api/v1/chat`：提供非流式问答接口，支持 `use_rag` 在知识库增强回答和普通直答之间切换。
- `POST /api/v1/chat/stream`：通过 SSE 推送模型生成片段，并在结束后写入会话历史。
- 会话管理：基于 `session_id` 在内存中维护多轮历史，滑动窗口保留最近 8 轮对话。

## API 示例

```http
POST /api/v1/chat/stream
Content-Type: application/json
```

```json
{
  "query": "向量数据库检索怎么优化？",
  "session_id": "demo-session",
  "use_rag": true
}
```

## 本地启动

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 环境变量

```text
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o
```
