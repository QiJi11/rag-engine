from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import LLM_MODEL, LLM_BASE_URL
from routers import chat, upload
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="RAG Engine",
    description="Knowledge QA demo with RAG and streaming responses",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(upload.router, prefix="/api/v1", tags=["Upload"])


@app.get("/health")
async def health():
    return {"status": "UP", "model": LLM_MODEL, "base_url": LLM_BASE_URL}

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
