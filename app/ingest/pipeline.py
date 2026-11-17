"""文档入库服务：新增、删除、重建。"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import EMBEDDING_DIM, RAW_DIR
from app.embeddings import BGEEmbedder
from app.ingest.loaders import SUPPORTED_SUFFIXES, iter_raw_documents, load_document
from app.ingest.splitter import split_document
from app.store.faiss_store import FaissStore

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    doc_name: str
    file_type: str
    chunks: int
    chars: int


class IngestionService:
    def __init__(
        self,
        store: FaissStore | None = None,
        embedder: BGEEmbedder | None = None,
    ) -> None:
        self.store = store or FaissStore()
        self._embedder = embedder

    @property
    def embedder(self) -> BGEEmbedder:
        if self._embedder is None:
            self._embedder = BGEEmbedder.get()
        return self._embedder

    def add_file(self, source: Path, copy_to_raw: bool = True) -> IngestResult:
        source = Path(source)
        suffix = source.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"不支持的文件类型：{source.suffix}")

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        target = RAW_DIR / source.name
        if copy_to_raw and source.resolve() != target.resolve():
            shutil.copy2(source, target)

        document = load_document(target)
        chunks = split_document(document)
        if not chunks:
            raise ValueError("文档没有可入库的内容")

        # 同名文档视为更新：先移除旧切片，再追加新向量
        if self.store.document_exists(document.doc_name):
            self.remove_document(document.doc_name, delete_raw=False)

        vectors = self.embedder.encode_documents(
            [chunk.content for chunk in chunks], show_progress=False
        )
        self.store.append_document(document, chunks, vectors)
        self.store.load()

        logger.info(
            "ingest add doc=%s type=%s chunks=%d chars=%d",
            document.doc_name,
            document.file_type,
            len(chunks),
            document.char_count,
        )
        return IngestResult(
            doc_name=document.doc_name,
            file_type=document.file_type,
            chunks=len(chunks),
            chars=document.char_count,
        )

    def remove_document(self, doc_name: str, delete_raw: bool = True) -> dict:
        record = self.store.get_document(doc_name)
        if record is None:
            raise ValueError(f"文档《{doc_name}》不存在")

        source_path = Path(record["source_path"])
        deleted_chunks = self.store.delete_document(doc_name)

        rows = self.store.chunk_rows()
        if rows:
            vectors = self.embedder.encode_documents(
                [row["content"] for row in rows], show_progress=False
            )
            self.store.reindex([row["id"] for row in rows], vectors)
        else:
            self.store.reindex([], np.empty((0, EMBEDDING_DIM), dtype="float32"))

        if delete_raw and source_path.exists() and RAW_DIR in source_path.parents:
            source_path.unlink()

        docs, chunks = self.store.stats()
        logger.info(
            "ingest remove doc=%s deleted_chunks=%d remaining_docs=%d remaining_chunks=%d",
            doc_name,
            deleted_chunks,
            docs,
            chunks,
        )
        return {
            "doc_name": doc_name,
            "deleted_chunks": deleted_chunks,
            "remaining_docs": docs,
            "remaining_chunks": chunks,
        }

    def rebuild(self) -> dict:
        paths = iter_raw_documents(RAW_DIR)
        if not paths:
            raise ValueError(f"{RAW_DIR} 下没有可用文档")

        docs = []
        chunks = []
        for path in paths:
            document = load_document(path)
            document_chunks = split_document(document)
            docs.append(document)
            chunks.extend(document_chunks)

        vectors = self.embedder.encode_documents(
            [chunk.content for chunk in chunks], show_progress=True
        )
        self.store.build(docs, chunks, vectors)

        logger.info("ingest rebuild docs=%d chunks=%d", len(docs), len(chunks))
        return {"docs": len(docs), "chunks": len(chunks)}
