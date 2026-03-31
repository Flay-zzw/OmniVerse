"""
TTS 服务 — 文字转语音（Kokoro，离线模型）
"""

import os
import re
import logging
import numpy as np
import soundfile as sf
from config import OUTPUT_DIR
from kokoro import KPipeline

logger = logging.getLogger("tts_service")

VOICE_ID = "zm_yunyang"
SPEED = 0.8
SAMPLE_RATE = 24000
SILENCE_DURATION = 0.35


def split_text_by_sentence(text: str) -> list[str]:
    sentences = re.split(r'(?<=[。！？!?…\n])|(?<=\.\.\.)', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    result = []
    for s in sentences:
        if len(s) > 100:
            sub = re.split(r'(?<=[，,；;])', s)
            result.extend(part.strip() for part in sub if part.strip())
        else:
            result.append(s)
    return result


class TTSService:
    def __init__(self):
        self.pipeline = None

    def load_model(self):
        logger.info("正在加载 Kokoro 模型...")
        self.pipeline = KPipeline(lang_code='z', repo_id='hexgrad/Kokoro-82M')
        logger.info("模型加载完成")

    def _synthesize_one(self, text: str) -> np.ndarray | None:
        try:
            audio_parts = [audio for _, _, audio in self.pipeline(text, voice=VOICE_ID, speed=SPEED)]
            if not audio_parts:
                logger.warning("空音频，跳过：%s...", text[:30])
                return None
            return np.concatenate(audio_parts)
        except Exception as e:
            logger.error("单句合成失败：%s... 错误：%s", text[:30], e)
            return None

    def text_to_speech(self, text: str, filename: str = None) -> str:
        if self.pipeline is None:
            self.load_model()

        if filename is None:
            filename = f"tts_{hash(text) % 100000}.wav"

        output_path = os.path.join(OUTPUT_DIR, filename)

        try:
            sentences = split_text_by_sentence(text)
            logger.info("文本共 %d 字，切分为 %d 句", len(text), len(sentences))

            all_audio = []
            silence = np.zeros(int(SAMPLE_RATE * SILENCE_DURATION), dtype=np.float32)

            for i, sentence in enumerate(sentences, 1):
                logger.info("  [%d/%d] (%d字) %s", i, len(sentences), len(sentence), sentence[:50])
                audio = self._synthesize_one(sentence)
                if audio is not None:
                    all_audio.append(audio)
                    if i < len(sentences):
                        all_audio.append(silence)

            if not all_audio:
                raise RuntimeError("所有句子都合成失败，请检查模型状态")

            full_audio = np.concatenate(all_audio)
            sf.write(output_path, full_audio, SAMPLE_RATE)
            logger.info("合成完成：%s（时长=%.1fs）", output_path, len(full_audio) / SAMPLE_RATE)

        except Exception as e:
            logger.exception("语音生成失败：%s", e)
            raise RuntimeError(f"语音生成失败：{str(e)}")

        return output_path