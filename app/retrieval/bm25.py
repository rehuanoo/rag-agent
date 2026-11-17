"""BM25 关键词检索：jieba 分词 + rank_bm25。"""
from __future__ import annotations

import logging
import re

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from app.config import BM25_TOP_K
from app.store.faiss_store import ChunkRecord, FaissStore

jieba.setLogLevel(logging.WARNING)

_VALID_TOKEN = re.compile(r"[\u4e00-\u9fa5A-Za-z0-9]")


def tokenize(text: str) -> list[str]:
    """搜索模式分词，保留中文、英文和数字，过滤标点与空白。"""
    tokens = []
    for token in jieba.lcut_for_search(text.lower()):
        token = token.strip()
        if token and _VALID_TOKEN.search(token):
            tokens.append(token)
    return tokens


class BM25Retriever:
    def __init__(self, store: FaissStore | None = None) -> None:
        self.store = store or FaissStore()
        self.store.load()
        self.chunks = self.store.all_chunks()
        self._corpus = [tokenize(chunk.content) for chunk in self.chunks]
        self.bm25 = BM25Okapi(self._corpus) if self._corpus else None

    def search(self, query: str, k: int = BM25_TOP_K) -> list[tuple[ChunkRecord, float]]:
        if not self.bm25 or not self.chunks:
            return []

        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [
            (self.chunks[index], float(scores[index]))
            for index in order
            if scores[index] > 0
        ]
