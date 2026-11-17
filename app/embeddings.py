"""本地 bge-large-zh-v1.5 向量化：transformers + mean pooling。

不依赖 sentence-transformers，避免其版本与模型旧配置不兼容的问题。
"""
from __future__ import annotations

import threading

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

torch.set_num_threads(1)

from app.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MAX_SEQ_LEN,
    EMBEDDING_MODEL_PATH,
)

# bge 官方建议：短查询转长文档检索时，给 query 加指令前缀，文档不加
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class BGEEmbedder:
    """对文档和查询统一做 L2 归一化，配合 FAISS 内积等价于余弦相似度。"""

    _instance: "BGEEmbedder | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        if not EMBEDDING_MODEL_PATH.exists():
            raise FileNotFoundError(f"未找到向量模型：{EMBEDDING_MODEL_PATH}")

        self.device = (
            EMBEDDING_DEVICE
            if EMBEDDING_DEVICE == "cpu" or torch.cuda.is_available()
            else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(EMBEDDING_MODEL_PATH))
        self.model = AutoModel.from_pretrained(str(EMBEDDING_MODEL_PATH))
        self.model.to(self.device).eval()

    @classmethod
    def get(cls) -> "BGEEmbedder":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _encode(self, texts: list[str], batch_size: int, show_progress: bool) -> np.ndarray:
        vectors: list[np.ndarray] = []
        batches = range(0, len(texts), batch_size)
        if show_progress:
            batches = tqdm(batches, desc="向量化", unit="batch")

        with torch.inference_mode():
            for start in batches:
                batch = texts[start : start + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=EMBEDDING_MAX_SEQ_LEN,
                    return_tensors="pt",
                ).to(self.device)

                last_hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(pooled.float().cpu().numpy())

        return np.concatenate(vectors, axis=0).astype("float32")

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        return self._encode(
            texts,
            batch_size or EMBEDDING_BATCH_SIZE,
            show_progress,
        )

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode(
            [QUERY_INSTRUCTION + text],
            batch_size=1,
            show_progress=False,
        )
