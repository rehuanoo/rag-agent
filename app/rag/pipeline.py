"""RAG 问答链路：混合检索 -> 组装上下文 -> 生成带引用的回答。"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.config import MAX_HISTORY_TURNS, TOP_K
from app.llm import chat_with_fallback
from app.rag.prompts import SYSTEM_PROMPT, build_user_prompt, format_context
from app.rag.query_rewrite import rewrite_query
from app.retrieval.hybrid import HybridRetriever, RetrievalHit, RetrievalResult
from app.store.faiss_store import FaissStore

logger = logging.getLogger(__name__)


@dataclass
class RagAnswer:
    answer: str
    hits: list[RetrievalHit] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    model: str = ""
    elapsed: float = 0.0
    retrieval_elapsed: float = 0.0
    rerank_elapsed: float = 0.0


def _format_context(hits: list[RetrievalHit]) -> str:
    return format_context(hits)


def _history_messages(history: list[dict] | None) -> list[BaseMessage]:
    if not history:
        return []

    messages: list[BaseMessage] = []
    for item in history[-MAX_HISTORY_TURNS * 2 :]:
        role = item.get("role")
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content[:500]))
        elif role == "assistant":
            messages.append(AIMessage(content=content[:500]))
    return messages


class RagPipeline:
    def __init__(self) -> None:
        self.store = FaissStore()
        self.ready = self.store.load()
        self.retriever: HybridRetriever | None = None
        if self.ready:
            self.retriever = HybridRetriever(store=self.store)

    def reload(self) -> bool:
        self.store = FaissStore()
        self.ready = self.store.load()
        self.retriever = HybridRetriever(store=self.store) if self.ready else None
        return self.ready

    def status_text(self) -> str:
        if not self.ready:
            return "索引未加载：请先运行 `python scripts/build_index.py`"
        docs, chunks = self.store.stats()
        return (
            f"**知识库已就绪**  \n"
            f"文档 {docs} 篇 · 切片 {chunks} 个  \n"
            f"检索：向量 + BM25 → RRF → bge-reranker-base  \n"
            f"向量模型 bge-large-zh-v1.5 · 向量库 FAISS"
        )

    def retrieve(
        self,
        question: str,
        use_rewrite: bool = False,
        use_rerank: bool = True,
        mode: str = "hybrid",
    ) -> RetrievalResult:
        if not self.ready or self.retriever is None:
            raise RuntimeError("索引尚未构建，请先运行 python scripts/build_index.py")

        extra_queries = rewrite_query(question)[1:] if use_rewrite else []
        return self.retriever.search(
            question,
            top_k=TOP_K,
            use_rerank=use_rerank,
            extra_queries=extra_queries,
            mode=mode,
        )

    def answer(
        self,
        question: str,
        history: list[dict] | None = None,
        use_rewrite: bool = False,
    ) -> RagAnswer:
        started = time.perf_counter()

        if not self.ready or self.retriever is None:
            return RagAnswer(
                answer="索引尚未构建。请先在项目根目录执行 `python scripts/build_index.py`，然后重载索引。",
                elapsed=time.perf_counter() - started,
            )

        retrieval = self.retrieve(question, use_rewrite=use_rewrite, use_rerank=True)
        hits = retrieval.hits
        if not hits:
            return RagAnswer(
                answer="知识库中没有检索到与该问题相关的内容。可以换一种说法，或补充更具体的制度名称。",
                queries=retrieval.queries,
                elapsed=time.perf_counter() - started,
                retrieval_elapsed=retrieval.elapsed,
            )

        context = _format_context(hits)
        prompt = build_user_prompt(question, context)
        messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
        messages.extend(_history_messages(history))
        messages.append(HumanMessage(content=prompt))

        answer_text, model = chat_with_fallback(messages)
        elapsed = time.perf_counter() - started
        logger.info(
            "rag answer question=%r hits=%d retrieval=%.2fs rerank=%.2fs model=%s total=%.2fs",
            question[:80],
            len(hits),
            retrieval.elapsed,
            retrieval.rerank_elapsed,
            model,
            elapsed,
        )
        return RagAnswer(
            answer=answer_text,
            hits=hits,
            queries=retrieval.queries,
            model=model,
            elapsed=elapsed,
            retrieval_elapsed=retrieval.elapsed,
            rerank_elapsed=retrieval.rerank_elapsed,
        )
