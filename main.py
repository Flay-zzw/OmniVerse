from fastapi import FastAPI
from routers import tts
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S"
)

# ==================== 创建 FastAPI 应用 ====================
app = FastAPI(
    title="万象 MyriadAI",
    description="多模态 AI 智能平台 — 听、说、看、写、译",
    version="1.0.0"
)

app.include_router(tts.router)