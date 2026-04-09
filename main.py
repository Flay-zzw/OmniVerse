import warnings
warnings.filterwarnings("ignore", message="torchcodec is not installed")

import sys
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import tts, chat, diarization, asr, ocr, translate
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S"
)


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


app = FastAPI(
    title="万象 MyriadAI",
    description="多模态 AI 智能平台 — 听、说、看、写、译",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(tts.router)
app.include_router(chat.router)
app.include_router(diarization.router)

app.include_router(asr.router)

app.include_router(ocr.router)
app.include_router(translate.router)
