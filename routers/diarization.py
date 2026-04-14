"""
Diarization 路由 — 说话人分割接口
"""

import os
import time
import asyncio
from fastapi import APIRouter, HTTPException, UploadFile, File
from config import OUTPUT_DIR
from services.diarization_service import DiarizationService

router = APIRouter(prefix="/diarization", tags=["说话人分割"])
diarization_service = DiarizationService()


@router.post("/analyze")
async def analyze_speakers(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        raise HTTPException(400, "仅支持 .wav 文件")

    save_path = os.path.join(OUTPUT_DIR, f"diarize_{int(time.time())}_{file.filename}")
    content = await file.read()
    try:
        with open(save_path, "wb") as f:
            f.write(content)
    finally:
        await file.close()

    try:
        start_time = time.time()
        # ← 用 asyncio.to_thread 包装阻塞调用
        segments = await asyncio.to_thread(diarization_service.diarize, save_path)

        return {
            "code": 200,
            "message": "说话人分割成功",
            "data": {
                "segments": segments,
                "processing_time_seconds": round(time.time() - start_time, 2),
            }
        }
    except Exception as e:
        raise HTTPException(500, f"说话人分割失败：{str(e)}")
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)