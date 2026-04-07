"""
ASR 路由 — 语音转文字
"""

import os
import uuid
import logging
import json
from fastapi import APIRouter, UploadFile, File, Form
from config import OUTPUT_DIR
from services.asr_service import ASRService
from services.diarization_service import DiarizationService
from services.chat_service import ChatService

diarization_service = DiarizationService()
chat_service = ChatService()

logger = logging.getLogger("router.asr")
router = APIRouter(prefix="/asr", tags=["ASR 语音转文字"])

asr_service = ASRService()


@router.post("/transcribe")
async def transcribe(
        file: UploadFile = File(...),
        language: str = Form("zh"),
):
    """上传音频文件，返回转写文本"""
    ext = os.path.splitext(file.filename)[1] or ".wav"
    tmp_name = f"asr_{uuid.uuid4().hex[:8]}{ext}"
    tmp_path = os.path.join(OUTPUT_DIR, tmp_name)

    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        result = asr_service.transcribe(tmp_path, language=language)
        return {
            "text": result.get("text", ""),
            "chunks": result.get("chunks", []),
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.post("/transcribe_with_speakers")
async def transcribe_with_speakers(
        file: UploadFile = File(...),
        language: str = Form("zh"),
):
    """上传音频 → 说话人分离 + 逐段转写"""
    ext = os.path.splitext(file.filename)[1] or ".wav"
    tmp_name = f"asr_diar_{uuid.uuid4().hex[:8]}{ext}"
    tmp_path = os.path.join(OUTPUT_DIR, tmp_name)

    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        # 1) 说话人分离
        logger.info("第一步：说话人分离")
        segments = diarization_service.diarize(tmp_path)

        # 2) 合并相邻同说话人片段（避免过短片段）
        merged = _merge_segments(segments, gap_threshold=0.5)

        # 3) 逐段转写
        logger.info("第二步：逐段 ASR，共 %d 段", len(merged))
        results = []
        for seg in merged:
            text = asr_service.transcribe_segment(
                tmp_path, seg["start"], seg["end"], language=language
            )
            results.append({
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "text": text,
            })

        #logger.info("第三步：Qwen 文字校对")
        #corrected_segments = _correct_text(results)

        logger.info("第四步：Qwen 会议总结")
        summary = _summarize_meeting(results)

        # return {"segments": results,
        #         "corrected_segments": corrected_segments,
        #         "summary": summary}
        return {"summary": summary}

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _merge_segments(segments: list[dict], gap_threshold: float = 0.5) -> list[dict]:
    """合并相邻且同一说话人的片段，间隔 < gap_threshold 秒则合并"""
    if not segments:
        return []

    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg["speaker"] == prev["speaker"] and (seg["start"] - prev["end"]) < gap_threshold:
            prev["end"] = seg["end"]
        else:
            merged.append(seg.copy())
    return merged


CORRECT_SYSTEM_PROMPT = """你是一个专业的中文语音识别校对助手。
用户会给你一段语音识别(ASR)的结果，其中可能存在错别字、同音替换、漏字、多字等问题。

你的任务：
1. 逐条校对每段文字，修正明显的ASR识别错误
2. 补充合理的标点符号
3. 不要改变原意，不要润色，不要添加原文没有的内容
4. 保持说话人的口语风格，不要改成书面语

请严格按照以下JSON格式返回，不要输出任何其他内容：
[
  {"speaker": "SPEAKER_XX", "text": "校对后的文字"},
  ...
]"""

SUMMARY_SYSTEM_PROMPT = """你是一个专业的会议纪要助手。
用户会给你一段会议的逐字稿（已标注说话人），请你生成一份结构清晰的会议总结。

要求：
1. 会议主题：一句话概括
2. 参会人员：列出所有说话人编号
3. 关键讨论点：按讨论顺序列出核心议题和各方观点
4. 决议事项：明确列出会议中达成的决定
5. 待办事项：列出需要跟进的任务，标注责任人
6. 风险提示：如有未解决的风险或分歧，列出

请严格按照以下JSON格式返回，不要输出任何其他内容：
{
    "meeting_topic": "一句话概括会议主题",
    "participants": ["SPEAKER_00", "SPEAKER_01"],
    "key_discussions": [
        {
            "topic": "讨论议题",
            "details": "各方观点和讨论内容"
        }
    ],
    "decisions": [
        "达成的决定1",
        "达成的决定2"
    ],
    "action_items": [
        {
            "task": "待办任务描述",
            "owner": "SPEAKER_XX"
        }
    ],
    "risks": [
        "未解决的风险或分歧"
    ]
}
"""


def _correct_text(segments: list[dict]) -> list[dict]:
    """调用 Qwen 对 ASR 结果做文字校对"""
    # 构造输入，只给 speaker + text
    input_data = [{"speaker": s["speaker"], "text": s["text"]} for s in segments]
    message = json.dumps(input_data, ensure_ascii=False)

    raw = chat_service.chat(message, system_prompt=CORRECT_SYSTEM_PROMPT)

    # 解析 Qwen 返回的 JSON
    try:
        # 兼容 Qwen 可能返回 ```json ... ``` 的情况
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        corrected = json.loads(cleaned)

        # 把校对后的 text 写回原始 segments（保留 start/end）
        for i, seg in enumerate(segments):
            if i < len(corrected):
                seg["text"] = corrected[i].get("text", seg["text"])
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning("Qwen 校对结果解析失败，使用原始文本：%s", e)

    return segments


def _summarize_meeting(segments: list[dict]) -> dict:
    """调用 Qwen 生成会议总结"""
    transcript_lines = []
    for seg in segments:
        transcript_lines.append(f"{seg['speaker']}：{seg['text']}")
    transcript = "\n".join(transcript_lines)

    raw = chat_service.chat(transcript, system_prompt=SUMMARY_SYSTEM_PROMPT)

    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        return json.loads(cleaned)
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("会议总结JSON解析失败，返回原始文本：%s", e)
        return {"raw_summary": raw}
