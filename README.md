# rag-engine

**企业级 RAG 知识库问答系统**

> 技术栈：Python + FastAPI + ChromaDB + SentenceTransformers + CrossEncoder + SSE 流式输出

## 项目架构

```
rag-engine/
├── main.py              # FastAPI 入口
├── config.py            # 配置（从 .env 读取）
├── routers/
│   ├── chat.py          # POST /api/v1/chat — 流式问答
│   └── upload.py        # POST /api/v1/upload — 文档上传
├── services/
│   ├── chunker.py       # 文档分块（512 chars + 50 overlap）
│   ├── llm.py           # LLM 调用（OpenAI-compatible，支持流式）
│   └── rag.py           # 两阶段检索（向量粗排 + CrossEncoder 精排）
└── store/
    └── vector_store.py  # ChromaDB 持久化向量存储
```

## 核心亮点（面试重点）

### 1. 真正的向量检索（不是 BM25）
- 使用 `all-MiniLM-L6-v2` 将文档映射到 384 维向量空间
- ChromaDB 持久化存储到 `./data/chroma_db`
- 余弦相似度检索，语义理解胜过关键词匹配

### 2. 两阶段 Rerank 重排
- **粗排**：向量检索 Top-15（快，召回率高）
- **精排**：CrossEncoder (`ms-marco-MiniLM-L-6-v2`) 对 15 个候选逐一打分 → 取 Top-3
- 精排比粗排准确 ~30%，代价是多一次推理

### 3. SSE 流式输出
- `chat_stream()` 使用 `stream=True` 逐 token 推送
- 前端实时显示，用户体验好

### 4. 可插拔向量存储
- 抽象了 `get_or_create_collection()` 接口
- MVP 用 ChromaDB，可无缝换成 Milvus/Weaviate/Qdrant

## 快速启动

```bash
# 1. 配置环境变量
cp .env.example .env  # 填入 LLM_API_KEY

# 2. 激活虚拟环境
.\venv\Scripts\activate  # Windows

# 3. 启动服务（端口 8000）
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API

```
GET  /health              # 健康检查
POST /api/v1/upload       # 上传文档（PDF/TXT）
POST /api/v1/chat         # 流式问答（SSE）
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_API_KEY` | API 密钥 | 必填 |
| `LLM_BASE_URL` | API 地址（OpenAI 兼容） | `http://143.198.212.179:18317/v1` |
| `LLM_MODEL` | 模型名 | `gpt-5.4` |

## 项目路径

```
D:\AtoC\dev\rag-engine
```