"""混合检索：向量 + BM25 -> RRF -> Cross-Encoder 精排。"""
from __future__ import annotations

import time
from dataclasses import dataclass

from app.config import (
    BM25_TOP_K,
    RERANK_CANDIDATES,
    RERANK_ENABLED,
    TOP_K,
)
from app.embeddings import BGEEmbedder
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import rrf_fuse
from app.retrieval.reranker import BGEReranker
from app.store.faiss_store import ChunkRecord, FaissStore


@dataclass
class RetrievalHit:
    chunk: ChunkRecord
    rrf_score: float = 0.0
    rerank_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None

    @property
    def doc_name(self) -> str:
        return self.chunk.doc_name

    @property
    def section(self) -> str:
        return self.chunk.section

    @property
    def page(self) -> int | None:
        return self.chunk.page

    @property
    def content(self) -> str:
        return self.chunk.content

    @property
    def citation(self) -> str:
        return self.chunk.citation

    @property
    def score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.rrf_score


@dataclass
class RetrievalResult:
    queries: list[str]
    hits: list[RetrievalHit]
    candidates: list[RetrievalHit]
    elapsed: float
    rerank_elapsed: float


class HybridRetriever:
    def __init__(
        self,
        store: FaissStore | None = None,
        embedder: BGEEmbedder | None = None,
        reranker: BGEReranker | None = None,
    ) -> None:
        self.store = store or FaissStore()
        self.store.load()
        self.embedder = embedder or BGEEmbedder.get()
        self.dense = DenseRetriever(store=self.store, embedder=self.embedder)
        self.bm25 = BM25Retriever(store=self.store)
        self.reranker = reranker

    def _get_reranker(self) -> BGEReranker:
        if self.reranker is None:
            self.reranker = BGEReranker.get()
        return self.reranker

    def warmup(self) -> None:
        """在主线程预加载向量和精排模型，避免首次请求在工作线程里初始化 CUDA。"""
        self.embedder.encode_query("预热")
        self._get_reranker()

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        use_rerank: bool = RERANK_ENABLED,
        extra_queries: list[str] | None = None,
        mode: str = "hybrid",
    ) -> RetrievalResult:
        started = time.perf_counter()
        queries = [query]
        for item in extra_queries or []:
            if item and item not in queries:
                queries.append(item)

        if mode == "dense":
            dense_hits = self.dense.search(query, k=top_k, min_score=0.0)
            hits = [
                RetrievalHit(chunk=hit, dense_rank=rank, dense_score=hit.score)
                for rank, hit in enumerate(dense_hits, start=1)
            ]
            return RetrievalResult(
                queries=queries,
                hits=hits,
                candidates=hits,
                elapsed=time.perf_counter() - started,
                rerank_elapsed=0.0,
            )

        ranked_lists: list[list[tuple[ChunkRecord, float]]] = []
        first_dense: list[tuple[ChunkRecord, float]] = []
        first_bm25: list[tuple[ChunkRecord, float]] = []

        for index, current_query in enumerate(queries):
            dense_pairs = [
                (hit, hit.score)
                for hit in self.dense.search(current_query, k=BM25_TOP_K, min_score=0.0)
            ]
            bm25_pairs = self.bm25.search(current_query, k=BM25_TOP_K)
            if index == 0:
                first_dense, first_bm25 = dense_pairs, bm25_pairs
            ranked_lists.extend([dense_pairs, bm25_pairs])

        fused = rrf_fuse(ranked_lists)
        dense_ranks = {chunk.faiss_pos: rank for rank, (chunk, _) in enumerate(first_dense, 1)}
        dense_scores = {chunk.faiss_pos: score for chunk, score in first_dense}
        bm25_ranks = {chunk.faiss_pos: rank for rank, (chunk, _) in enumerate(first_bm25, 1)}
        bm25_scores = {chunk.faiss_pos: score for chunk, score in first_bm25}

        candidates = [
            RetrievalHit(
                chunk=item.chunk,
                rrf_score=item.rrf_score,
                dense_rank=dense_ranks.get(item.chunk.faiss_pos),
                dense_score=dense_scores.get(item.chunk.faiss_pos),
                bm25_rank=bm25_ranks.get(item.chunk.faiss_pos),
                bm25_score=bm25_scores.get(item.chunk.faiss_pos),
            )
            for item in fused
        ]

        rerank_elapsed = 0.0
        if use_rerank and candidates:
            pool = candidates[:RERANK_CANDIDATES]
            rerank_started = time.perf_counter()
            scores = self._get_reranker().score(
                query, [item.chunk.content for item in pool]
            )
            rerank_elapsed = time.perf_counter() - rerank_started
            for item, score in zip(pool, scores):
                item.rerank_score = score
            pool.sort(key=lambda item: item.rerank_score or 0.0, reverse=True)
            final_hits = pool[:top_k]
        else:
            final_hits = candidates[:top_k]

        return RetrievalResult(
            queries=queries,
            hits=final_hits,
            candidates=candidates,
            elapsed=time.perf_counter() - started,
            rerank_elapsed=rerank_elapsed,
        )
