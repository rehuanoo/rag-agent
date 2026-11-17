"""本地 Qwen1.5-1.8B 兜底生成，用于远端 API 限流时保持可用。"""
from __future__ import annotations

import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import (
    LOCAL_LLM_MAX_INPUT_TOKENS,
    LOCAL_LLM_MAX_NEW_TOKENS,
    LOCAL_LLM_PATH,
)


class LocalChatModel:
    _instance: "LocalChatModel | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        if not LOCAL_LLM_PATH.exists():
            raise FileNotFoundError(f"未找到本地模型：{LOCAL_LLM_PATH}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_LLM_PATH))
        self.model = AutoModelForCausalLM.from_pretrained(
            str(LOCAL_LLM_PATH),
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )
        self.model.to(self.device).eval()

    @classmethod
    def get(cls) -> "LocalChatModel":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def chat(self, messages: list[dict]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            truncation=True,
            max_length=LOCAL_LLM_MAX_INPUT_TOKENS,
        ).to(self.device)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=LOCAL_LLM_MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
