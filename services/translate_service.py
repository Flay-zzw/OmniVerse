"""
Translate 服务 — 文本翻译（NLLB-200-distilled-600M，离线模型）
"""

import re
import logging
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger("translate_service")

MODEL_NAME = "facebook/nllb-200-distilled-600M"

# NLLB 语言代码映射
LANG_CODE_MAP = {
    "zh": "zho_Hans",   # 简体中文
    "en": "eng_Latn",   # 英语
}


def _split_sentences(text: str, lang: str) -> list[str]:
    """
    按句子切分文本，避免单次输入过长导致翻译截断
    """
    if lang == "zh":
        # 中文按常见句末标点切分
        parts = re.split(r'(?<=[。！？!?\n])', text)
    else:
        # 英文按句号、感叹号、问号切分
        parts = re.split(r'(?<=[.!?\n])\s*', text)

    return [p.strip() for p in parts if p.strip()]


class TranslateService:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_model(self):
        logger.info("正在加载 %s ...", MODEL_NAME)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model.to("cpu")

        logger.info("模型加载完成")

    def _translate_single(self, text: str, src_code: str, tgt_code: str) -> str:
        """翻译单个句子"""
        self.tokenizer.src_lang = src_code

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = inputs.to(self.model.device)

        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_code)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=512,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def translate(self, text: str, source_lang: str = "zh", target_lang: str = "en") -> str:
        """
        翻译文本（自动分句，逐句翻译后拼接）
        :param text: 待翻译文本
        :param source_lang: 源语言代码（zh / en）
        :param target_lang: 目标语言代码（zh / en）
        :return: 翻译后的文本
        """
        if self.model is None:
            self.load_model()

        src_code = LANG_CODE_MAP.get(source_lang)
        tgt_code = LANG_CODE_MAP.get(target_lang)

        if src_code is None:
            raise ValueError(f"不支持的源语言：{source_lang}，支持：{list(LANG_CODE_MAP.keys())}")
        if tgt_code is None:
            raise ValueError(f"不支持的目标语言：{target_lang}，支持：{list(LANG_CODE_MAP.keys())}")
        if source_lang == target_lang:
            raise ValueError("源语言和目标语言不能相同")

        logger.info("开始翻译：%s → %s，文本长度 %d 字", source_lang, target_lang, len(text))

        # 分句翻译，避免长文本截断
        sentences = _split_sentences(text, source_lang)
        logger.info("文本切分为 %d 句", len(sentences))

        translated_parts = []
        for i, sentence in enumerate(sentences, 1):
            logger.info("  [%d/%d] %s", i, len(sentences), sentence[:50])
            result = self._translate_single(sentence, src_code, tgt_code)
            translated_parts.append(result)

        # 拼接：英文用空格，中文直接拼
        if target_lang == "en":
            full_result = " ".join(translated_parts)
        else:
            full_result = "".join(translated_parts)

        logger.info("翻译完成：%s", full_result[:100])
        return full_result