"""
Diarization 服务 — 说话人分割（pyannote 3.0, 离线模型）
"""

import os
import logging
import torchaudio
from pyannote.audio import Pipeline

logger = logging.getLogger("diarization_service")

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_NAME = "pyannote/speaker-diarization-3.0"


class DiarizationService:
    def __init__(self):
        self.pipeline = None

    def load_model(self):
        logger.info("正在加载 %s ...", MODEL_NAME)

        # hf-mirror 不支持 gated model，临时切回官方源下载
        old_endpoint = os.environ.get("HF_ENDPOINT", "")
        # os.environ["HF_ENDPOINT"] = "https://huggingface.co"

        try:
            self.pipeline = Pipeline.from_pretrained(
                MODEL_NAME,
                use_auth_token=HF_TOKEN,
            )
        finally:
            if old_endpoint:
                os.environ["HF_ENDPOINT"] = old_endpoint
            else:
                os.environ.pop("HF_ENDPOINT", None)

        logger.info("模型加载完成")

    def diarize(self, audio_path: str) -> list[dict]:
        if self.pipeline is None:
            self.load_model()

        logger.info("开始分割：%s", audio_path)

        waveform, sample_rate = torchaudio.load(audio_path)
        diarization = self.pipeline({"waveform": waveform, "sample_rate": sample_rate})

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "speaker": speaker,
            })

        logger.info("分割完成，共 %d 个片段", len(segments))
        return segments