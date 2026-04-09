"""
Translate 路由 — 文本翻译接口
"""

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.translate_service import TranslateService

router = APIRouter(prefix="/translate", tags=["文本翻译"])
translate_service = TranslateService()


class TranslateRequest(BaseModel):
    text: str
    source_lang: str = "zh"    # zh / en
    target_lang: str = "en"    # zh / en


@router.post("/text")
def translate_text(request: TranslateRequest):        # ← 改为 def
    if not request.text.strip():
        raise HTTPException(400, "翻译内容不能为空")

    try:
        start_time = time.time()
        result = translate_service.translate(
            request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
        )

        return {
            "code": 200,
            "message": "翻译成功",
            "data": {
                "original": request.text,
                "translated": result,
                "source_lang": request.source_lang,
                "target_lang": request.target_lang,
                "processing_time_seconds": round(time.time() - start_time, 2),
            }
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"翻译失败：{str(e)}")