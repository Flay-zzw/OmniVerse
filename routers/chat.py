"""
Chat 路由 — 文本对话接口
"""

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["文本对话"])
chat_service = ChatService()


class ChatRequest(BaseModel):
    message: str
    system_prompt: str | None = None  # 可选，不传就用默认值


@router.post("/completions")
def chat_completions(request: ChatRequest):          # ← 改为 def，FastAPI 自动放线程池
    if not request.message.strip():
        raise HTTPException(400, "消息内容不能为空")

    try:
        start_time = time.time()
        content = chat_service.chat(request.message, request.system_prompt)

        return {
            "code": 200,
            "message": "对话成功",
            "data": {
                "content": content,
                "processing_time_seconds": round(time.time() - start_time, 2),
            }
        }
    except Exception as e:
        raise HTTPException(500, f"对话失败：{str(e)}")