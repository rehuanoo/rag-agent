"""RRF（Reciprocal Rank Fusion）融合多路召回结果。"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import RRF_K
from app.store.faiss_store import ChunkRecord


@dataclass
class FusedHit:
    chunk: ChunkRecord
    rrf_score: float = 0.0
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None

    @property
    def score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.rrf_score


def rrf_fuse(
    ranked_lists: list[list[tuple[ChunkRecord, float]]],
    k: int = RRF_K,
) -> list[FusedHit]:
    """每一路按名次贡献 1/(k+rank)，避免不同检索器的分数量纲不一致。"""
    fused: dict[int, FusedHit] = {}

    for ranked in ranked_lists:
        for rank, (chunk, _) in enumerate(ranked, start=1):
            hit = fused.get(chunk.faiss_pos)
            if hit is None:
                hit = FusedHit(chunk=chunk)
                fused[chunk.faiss_pos] = hit
            hit.rrf_score += 1.0 / (k + rank)

    return sorted(fused.values(), key=lambda item: item.rrf_score, reverse=True)
