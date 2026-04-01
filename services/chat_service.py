"""
Chat 服务 — 文本对话（Qwen3-0.6B，离线模型）
"""

import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("chat_service")

MODEL_NAME = "Qwen/Qwen3-0.6B"


class ChatService:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_model(self):
        logger.info("正在加载 Qwen3-0.6B 模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto"
        )
        logger.info("模型加载完成")

    def chat(self, message: str, system_prompt: str = None) -> str:
        if self.model is None:
            self.load_model()

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages = [{"role": "user", "content": message}]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking = False,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        output = self.model.generate(**inputs, max_new_tokens=2048)
        output_ids = output[0][len(inputs.input_ids[0]):]

        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()