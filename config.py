import os
from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://143.198.212.179:18317/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.4-mini")
