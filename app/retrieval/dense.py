"""稠密向量检索。"""
from __future__ import annotations

from app.config import MIN_SCORE, TOP_K
from app.embeddings import BGEEmbedder
from app.store.faiss_store import ChunkRecord, FaissStore


class DenseRetriever:
    def __init__(
        self,
        store: FaissStore | None = None,
        embedder: BGEEmbedder | None = None,
    ) -> None:
        self.store = store or FaissStore()
        self.embedder = embedder or BGEEmbedder.get()
        self.store.load()

    def search(
        self,
        query: str,
        k: int = TOP_K,
        min_score: float = MIN_SCORE,
    ) -> list[ChunkRecord]:
        query_vector = self.embedder.encode_query(query)
        hits = self.store.search(query_vector, k)
        filtered = [hit for hit in hits if hit.score >= min_score]
        # 全部低于阈值时保留最好的几条，交给提示词判断“资料不足”
        return filtered or hits[: max(1, k // 2)]
