"""bge-reranker-base 精排：Cross-Encoder 逐对打分。"""
from __future__ import annotations

import threading

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import RERANK_BATCH_SIZE, RERANKER_MODEL_PATH


class BGEReranker:
    _instance: "BGEReranker | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        if not RERANKER_MODEL_PATH.exists():
            raise FileNotFoundError(f"未找到精排模型：{RERANKER_MODEL_PATH}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(str(RERANKER_MODEL_PATH))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(RERANKER_MODEL_PATH),
            dtype=dtype,
        )
        self.model.to(self.device).eval()

    @classmethod
    def get(cls) -> "BGEReranker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def score(self, query: str, texts: list[str], batch_size: int = RERANK_BATCH_SIZE) -> list[float]:
        if not texts:
            return []

        scores: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                inputs = self.tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**inputs).logits.view(-1).float()
                scores.extend(torch.sigmoid(logits).cpu().tolist())
        return scores
