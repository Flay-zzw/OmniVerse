"""
TTS 路由 — 文字转语音接口
"""

import os
import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from config import OUTPUT_DIR
from services.tts_service import TTSService

router = APIRouter(prefix="/tts", tags=["文字转语音"])
tts_service = TTSService()


class TTSRequest(BaseModel):
    text: str


@router.post("/generate")
def generate_speech(request: TTSRequest):             # ← 改为 def
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


@router.get("/download/{filename}")
async def download_audio(filename: str):              # ← 这个保持 async 没问题，无阻塞
    """下载生成的语音文件"""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(400, "非法文件名")

    file_path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(404, "文件不存在或已过期")

    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        filename=filename,
    )