"""
OCR 服务 — 图片文字提取（Qwen2-VL-2B-Instruct，离线模型）
"""

import logging
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

logger = logging.getLogger("ocr_service")

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

# 限制视觉 token 数量，大幅提升 CPU 推理速度
# 默认范围是 4~16384 个 token，这里限制到 256~768
# 对 OCR 文字提取来说足够了
MIN_PIXELS = 256 * 28 * 28    # 200704
MAX_PIXELS = 768 * 28 * 28    # 602112

OCR_PROMPT = "请提取这张图片中的所有文字内容，保持原始排版格式，只输出文字，不要添加任何解释。"


class OCRService:
    def __init__(self):
        self.model = None
        self.processor = None

    def load_model(self):
        logger.info("正在加载 %s ...", MODEL_NAME)

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )

        logger.info("模型加载完成")

    def extract_text(self, image_path: str, prompt: str = None) -> str:
        """
        从图片中提取文字
        :param image_path: 图片文件路径
        :param prompt: 自定义提示词，默认使用 OCR 提取提示
        :return: 提取到的文字内容
        """
        if self.model is None:
            self.load_model()

        prompt = prompt or OCR_PROMPT
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_input = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text_input], images=[image], padding=True, return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=2048)
        # 截掉输入部分，只保留生成内容
        generated_ids = output_ids[0][len(inputs.input_ids[0]):]
        result = self.processor.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        logger.info("OCR 完成，提取文字 %d 字", len(result))
        return result.strip()