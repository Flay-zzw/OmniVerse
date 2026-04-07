"""
ASR 服务 — 语音转文字（Whisper large-v3-turbo, 离线模型）
"""

import logging
import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

logger = logging.getLogger("asr_service")

MODEL_NAME = "openai/whisper-large-v3-turbo"


class ASRService:
    def __init__(self):
        self.pipe = None

    def load_model(self):
        logger.info("正在加载 %s ...", MODEL_NAME)

        device = "cpu"
        torch_dtype = torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(MODEL_NAME)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

        logger.info("模型加载完成")

    def transcribe(self, audio_path: str, language: str = "zh") -> dict:
        """
        整段音频转文字
        返回: {"text": "...", "chunks": [{"timestamp": (start, end), "text": "..."}]}
        """
        if self.pipe is None:
            self.load_model()

        logger.info("开始转写：%s", audio_path)

        result = self.pipe(
            audio_path,
            generate_kwargs={"language": language},
            return_timestamps=True,
        )

        logger.info("转写完成，共 %d 字", len(result.get("text", "")))
        return result

    def transcribe_segment(self, audio_path: str, start: float, end: float,
                           language: str = "zh") -> str:
        """
        对指定时间段做转写（配合 diarization 使用）
        """
        if self.pipe is None:
            self.load_model()

        waveform, sample_rate = torchaudio.load(audio_path)

        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segment_waveform = waveform[:, start_sample:end_sample]

        # pipeline 接收 numpy 数组（单声道, float32）
        if segment_waveform.shape[0] > 1:
            segment_waveform = segment_waveform.mean(dim=0)
        else:
            segment_waveform = segment_waveform.squeeze(0)

        audio_numpy = segment_waveform.numpy()

        result = self.pipe(
            {"array": audio_numpy, "sampling_rate": sample_rate},
            generate_kwargs={"language": language},
        )

        return result.get("text", "").strip()