"""
OCR 路由 — 图片文字提取
"""

import os
import uuid
import time
import logging
from fastapi import APIRouter, UploadFile, File, Form
from config import OUTPUT_DIR
from services.ocr_service import OCRService

logger = logging.getLogger("router.ocr")
router = APIRouter(prefix="/ocr", tags=["OCR 图片文字提取"])

ocr_service = OCRService()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


@router.post("/extract")
async def extract_text(
        file: UploadFile = File(...),
        prompt: str = Form(None),
):
    """上传图片，返回提取的文字内容"""
    ext = os.path.splitext(file.filename)[1].lower() or ".png"
    if ext not in ALLOWED_EXTENSIONS:
        from fastapi import HTTPException
        raise HTTPException(400, f"不支持的图片格式：{ext}，仅支持 {', '.join(ALLOWED_EXTENSIONS)}")

    tmp_name = f"ocr_{uuid.uuid4().hex[:8]}{ext}"
    tmp_path = os.path.join(OUTPUT_DIR, tmp_name)

    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        start_time = time.time()
        text = ocr_service.extract_text(tmp_path, prompt=prompt)

        return {
            "code": 200,
            "message": "文字提取成功",
            "data": {
                "text": text,
                "processing_time_seconds": round(time.time() - start_time, 2),
            }
        }
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(500, f"文字提取失败：{str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)