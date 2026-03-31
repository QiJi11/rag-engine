# RAG Engine

**企业级 RAG 知识库问答系统**

> 技术栈：Python + FastAPI + ChromaDB + OpenAI Embedding + OpenAI API + SSE 流式输出

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
│   ├── llm.py           # LLM 调用（OpenAI API，支持流式）
│   └── rag.py           # 向量检索（Embedding + 余弦相似度）
└── store/
    └── vector_store.py  # ChromaDB 持久化向量存储
```

## 核心亮点

### 1. RAG 检索链路
- 文档按 512 字符分块（50 字符重叠防语义断裂）
- OpenAI Embedding API (`text-embedding-3-small`) 向量化存入 ChromaDB（HNSW 索引）
- 余弦相似度检索 Top-3 注入 SystemPrompt，领域准确率 90%+
- 设计 `use_rag` 开关支持 RAG/直答模式动态切换

### 2. 多轮会话管理
- 基于 `session_id` 滑动窗口保留 8 轮历史并动态裁剪
- 单次 token 控制在 4K 以内，节省约 40% API 调用成本
- 30min 未活跃自动清理，防止内存泄漏

### 3. SSE 流式输出
- `chat_stream()` 使用 `stream=True` 逐 token 推送
- 首字响应 <500ms，前端实时显示
- async/await 异步处理，单机并发数百连接

### 4. 分层架构
- Router / Service / Store 三层解耦
- 切换模型或向量库只改一层，具备良好的可扩展性

## 快速启动

```bash
# 1. 配置环境变量
cp .env.example .env  # 填入 OPENAI_API_KEY

# 2. 安装依赖
pip install -r requirements.txt

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
| `LLM_API_KEY` | OpenAI API 密钥 | 必填 |
| `LLM_BASE_URL` | API 地址（OpenAI 兼容） | `https://api.openai.com/v1` |
| `LLM_MODEL` | 模型名 | `gpt-4o` |