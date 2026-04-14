"""
ASR 路由 — 语音转文字
"""

import os
import uuid
import asyncio
import logging
import json
import subprocess
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from config import OUTPUT_DIR
from services.asr_service import ASRService
from services.diarization_service import DiarizationService
from services.chat_service import ChatService

diarization_service = DiarizationService()
chat_service = ChatService()

logger = logging.getLogger("router.asr")
router = APIRouter(prefix="/asr", tags=["ASR 语音转文字"])

asr_service = ASRService()

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma", ".aac"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".ts"}
ALLOWED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


async def _save_upload(file: UploadFile, prefix: str) -> tuple[str, str]:
    """保存上传文件，必要时转为 wav，返回 (原始路径, wav路径)"""
    ext = os.path.splitext(file.filename)[1].lower() or ".wav"
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"不支持的文件格式：{ext}")

    tmp_name = f"{prefix}_{uuid.uuid4().hex[:8]}{ext}"
    tmp_path = os.path.join(OUTPUT_DIR, tmp_name)

    content = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)

    wav_path = await asyncio.to_thread(_ensure_wav, tmp_path)
    return tmp_path, wav_path


def _cleanup(*paths):
    """清理临时文件"""
    seen = set()
    for p in paths:
        if p and p not in seen and os.path.exists(p):
            os.remove(p)
            seen.add(p)

def _ensure_wav(file_path: str) -> str:
    """如果是视频文件，用 ffmpeg 提取音频并转为 wav；音频文件直接返回原路径"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in AUDIO_EXTENSIONS:
        return file_path

    wav_path = file_path.rsplit(".", 1)[0] + ".wav"
    logger.info("检测到视频文件，正在用 ffmpeg 提取音频：%s → %s", file_path, wav_path)

    cmd = [
        "ffmpeg", "-i", file_path,
        "-vn",                # 去掉视频流
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", "16000",       # 16kHz 采样率（Whisper 推荐）
        "-ac", "1",           # 单声道
        "-y",                 # 覆盖已有文件
        wav_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("ffmpeg 失败：%s", result.stderr)
        raise RuntimeError(f"ffmpeg 音频提取失败：{result.stderr[:200]}")

    logger.info("音频提取完成：%s", wav_path)
    return wav_path



@router.post("/transcribe")
async def transcribe(
        file: UploadFile = File(...),
        language: str = Form("zh"),
):
    tmp_path, wav_path = await _save_upload(file, "asr")
    try:
        result = await asyncio.to_thread(asr_service.transcribe, wav_path, language)
        return {
            "text": result.get("text", ""),
            "chunks": result.get("chunks", []),
        }
    finally:
        _cleanup(tmp_path, wav_path)


@router.post("/transcribe_with_speakers")
async def transcribe_with_speakers(
        file: UploadFile = File(...),
        language: str = Form("zh"),
):
    tmp_path, wav_path = await _save_upload(file, "asr_diar")
    try:
        result = await asyncio.to_thread(
            _do_transcribe_with_speakers, wav_path, language
        )
        return result
    finally:
        _cleanup(tmp_path, wav_path)


def _do_transcribe_with_speakers(tmp_path: str, language: str) -> dict:
    """同步执行：说话人分离 + 逐段转写 + 会议总结（在线程中运行）"""
    # 1) 说话人分离
    logger.info("第一步：说话人分离")
    segments = diarization_service.diarize(tmp_path)

    # 2) 合并相邻同说话人片段
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

    # 4) 会议总结
    logger.info("第四步：Qwen 会议总结")
    summary = _summarize_meeting(results)

    return {"summary": summary}


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