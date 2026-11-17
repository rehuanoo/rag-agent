"""构建 FAISS 索引：解析 -> 切分 -> 向量化 -> 落盘。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import RAW_DIR
from app.embeddings import BGEEmbedder
from app.ingest.loaders import iter_raw_documents, load_document
from app.ingest.splitter import split_document
from app.logging_utils import setup_logging
from app.store.faiss_store import FaissStore


def main() -> None:
    setup_logging()
    paths = iter_raw_documents(RAW_DIR)
    if not paths:
        raise SystemExit(f"未在 {RAW_DIR} 找到文档，请先运行 python scripts/generate_docs.py")

    started = time.perf_counter()
    docs = []
    chunks = []

    print(f"发现 {len(paths)} 个文档\n")
    for path in paths:
        document = load_document(path)
        document_chunks = split_document(document)
        docs.append(document)
        chunks.extend(document_chunks)
        print(
            f"{path.name:<32} 类型 {document.file_type:<9} "
            f"章节 {len(document.sections):>3}  切片 {len(document_chunks):>3}"
        )

    print(f"\n共 {len(chunks)} 个切片，开始向量化（{BGEEmbedder.get().model.device}）…")
    embedder = BGEEmbedder.get()
    vectors = embedder.encode_documents([chunk.content for chunk in chunks])

    store = FaissStore()
    store.build(docs, chunks, vectors)
    doc_count, chunk_count = store.stats()

    print(
        f"索引构建完成：{doc_count} 篇文档，{chunk_count} 个切片，"
        f"向量维度 {vectors.shape[1]}，耗时 {time.perf_counter() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
