"""
TTS 路由 — 文字转语音接口
"""

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.tts_service import TTSService

router = APIRouter(prefix="/tts", tags=["文字转语音"])
tts_service = TTSService()


class TTSRequest(BaseModel):
    text: str


@router.post("/generate")
async def generate_speech(request: TTSRequest):
    if not request.text.strip():
        raise HTTPException(400, "文字内容不能为空")

    try:
        start_time = time.time()
        filename = f"tts_{int(time.time())}.wav"
        tts_service.text_to_speech(request.text, filename)

        return {
            "code": 200,
            "message": "语音生成成功",
            "data": {
                "file_url": f"/tts/download/{filename}",
                "text_length": len(request.text),
                "processing_time_seconds": round(time.time() - start_time, 2),
            }
        }
    except Exception as e:
        raise HTTPException(500, f"语音生成失败：{str(e)}")