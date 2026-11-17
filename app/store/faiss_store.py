"""FAISS 向量索引 + SQLite 元数据存储。"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np

faiss.omp_set_num_threads(1)

from app.config import EMBEDDING_DIM, FAISS_INDEX_PATH, META_DB_PATH
from app.ingest.loaders import LoadedDoc
from app.ingest.splitter import Chunk


@dataclass
class ChunkRecord:
    doc_id: str
    doc_name: str
    section: str
    page: int | None
    chunk_index: int
    content: str
    faiss_pos: int
    score: float = 0.0

    @property
    def citation(self) -> str:
        page = f" · 第 {self.page} 页" if self.page else ""
        return f"《{self.doc_name}》/ {self.section}{page}"


class FaissStore:
    def __init__(self) -> None:
        self.index: faiss.Index | None = None

    # ---------- 建库 ----------
    def build(self, docs: list[LoadedDoc], chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunk 数量与向量数量不一致")
        if vectors.shape[1] != EMBEDDING_DIM:
            raise ValueError(f"向量维度 {vectors.shape[1]} 与配置 {EMBEDDING_DIM} 不一致")

        Path(FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
        if META_DB_PATH.exists():
            META_DB_PATH.unlink()

        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(np.ascontiguousarray(vectors, dtype="float32"))
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        self.index = index

        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            self._init_schema(conn)

            chunk_counts: dict[str, int] = {}
            for chunk in chunks:
                chunk_counts[chunk.doc_id] = chunk_counts.get(chunk.doc_id, 0) + 1

            conn.executemany(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        doc.doc_id,
                        doc.doc_name,
                        doc.file_type,
                        doc.source_path,
                        doc.char_count,
                        chunk_counts.get(doc.doc_id, 0),
                        now,
                    )
                    for doc in docs
                ],
            )
            conn.executemany(
                """
                INSERT INTO chunks
                    (doc_id, doc_name, section, page, chunk_index, content, faiss_pos)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.doc_id,
                        chunk.doc_name,
                        chunk.section,
                        chunk.page,
                        chunk.chunk_index,
                        chunk.content,
                        pos,
                    )
                    for pos, chunk in enumerate(chunks)
                ],
            )

    def append_document(
        self,
        doc: LoadedDoc,
        chunks: list[Chunk],
        vectors: np.ndarray,
    ) -> None:
        """增量追加一篇文档：只新增向量，不重建已有索引。"""
        if not chunks:
            raise ValueError("文档没有可入库的切片")
        if len(chunks) != len(vectors):
            raise ValueError("chunk 数量与向量数量不一致")
        if self.document_exists(doc.doc_name):
            raise ValueError(f"文档《{doc.doc_name}》已存在，请先删除再新增")

        Path(FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
        if FAISS_INDEX_PATH.exists():
            index = faiss.read_index(str(FAISS_INDEX_PATH))
        else:
            index = faiss.IndexFlatIP(EMBEDDING_DIM)

        offset = index.ntotal
        index.add(np.ascontiguousarray(vectors, dtype="float32"))
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        self.index = index

        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            self._init_schema(conn)
            conn.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    doc.doc_id,
                    doc.doc_name,
                    doc.file_type,
                    doc.source_path,
                    doc.char_count,
                    len(chunks),
                    now,
                ),
            )
            conn.executemany(
                """
                INSERT INTO chunks
                    (doc_id, doc_name, section, page, chunk_index, content, faiss_pos)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.doc_id,
                        chunk.doc_name,
                        chunk.section,
                        chunk.page,
                        chunk.chunk_index,
                        chunk.content,
                        offset + position,
                    )
                    for position, chunk in enumerate(chunks)
                ],
            )

    def reindex(self, chunk_ids: list[int], vectors: np.ndarray) -> None:
        """按给定的 chunk id 顺序重建索引，用于删除文档后重排位置。"""
        if len(chunk_ids) != len(vectors):
            raise ValueError("chunk id 数量与向量数量不一致")

        if not chunk_ids:
            if FAISS_INDEX_PATH.exists():
                FAISS_INDEX_PATH.unlink()
            self.index = None
            return

        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(np.ascontiguousarray(vectors, dtype="float32"))
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        self.index = index

        with self._connect() as conn:
            conn.executemany(
                "UPDATE chunks SET faiss_pos = ? WHERE id = ?",
                [(position, chunk_id) for position, chunk_id in enumerate(chunk_ids)],
            )

    # ---------- 文档管理 ----------
    def document_exists(self, doc_name: str) -> bool:
        if not META_DB_PATH.exists():
            return False
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM documents WHERE doc_name = ? LIMIT 1", (doc_name,)
            ).fetchone()
        return row is not None

    def get_document(self, doc_name: str) -> sqlite3.Row | None:
        if not META_DB_PATH.exists():
            return None
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM documents WHERE doc_name = ?", (doc_name,)
            ).fetchone()

    def delete_document(self, doc_name: str) -> int:
        if not META_DB_PATH.exists():
            return 0
        with self._connect() as conn:
            deleted = conn.execute(
                "DELETE FROM chunks WHERE doc_name = ?", (doc_name,)
            ).rowcount
            conn.execute("DELETE FROM documents WHERE doc_name = ?", (doc_name,))
        return int(deleted)

    def chunk_rows(self) -> list[sqlite3.Row]:
        if not META_DB_PATH.exists():
            return []
        with self._connect() as conn:
            return conn.execute(
                "SELECT id, content FROM chunks ORDER BY faiss_pos"
            ).fetchall()

    # ---------- 查询 ----------
    def load(self) -> bool:
        if not FAISS_INDEX_PATH.exists() or not META_DB_PATH.exists():
            return False
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        return True

    def search(self, query_vector: np.ndarray, k: int) -> list[ChunkRecord]:
        if self.index is None and not self.load():
            return []

        k = min(k, self.index.ntotal)
        if k <= 0:
            return []

        scores, positions = self.index.search(
            np.ascontiguousarray(query_vector, dtype="float32"), k
        )
        positions = [int(pos) for pos in positions[0] if pos >= 0]
        if not positions:
            return []

        placeholders = ",".join("?" for _ in positions)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE faiss_pos IN ({placeholders})",
                positions,
            ).fetchall()

        by_position = {row["faiss_pos"]: row for row in rows}
        results: list[ChunkRecord] = []
        for rank, position in enumerate(positions):
            row = by_position.get(position)
            if row is None:
                continue
            results.append(
                ChunkRecord(
                    doc_id=row["doc_id"],
                    doc_name=row["doc_name"],
                    section=row["section"],
                    page=row["page"],
                    chunk_index=row["chunk_index"],
                    content=row["content"],
                    faiss_pos=position,
                    score=float(scores[0][rank]),
                )
            )
        return results

    def list_documents(self) -> list[sqlite3.Row]:
        if not META_DB_PATH.exists():
            return []
        with self._connect() as conn:
            return conn.execute(
                "SELECT doc_name, file_type, chunk_count, char_count FROM documents ORDER BY doc_name"
            ).fetchall()

    def all_chunks(self) -> list[ChunkRecord]:
        if not META_DB_PATH.exists():
            return []
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM chunks ORDER BY faiss_pos").fetchall()
        return [
            ChunkRecord(
                doc_id=row["doc_id"],
                doc_name=row["doc_name"],
                section=row["section"],
                page=row["page"],
                chunk_index=row["chunk_index"],
                content=row["content"],
                faiss_pos=row["faiss_pos"],
            )
            for row in rows
        ]

    def stats(self) -> tuple[int, int]:
        if not META_DB_PATH.exists():
            return 0, 0
        with self._connect() as conn:
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return int(doc_count), int(chunk_count)

    @staticmethod
    def _connect() -> sqlite3.Connection:
        conn = sqlite3.connect(META_DB_PATH, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _init_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                doc_name TEXT NOT NULL,
                file_type TEXT NOT NULL,
                source_path TEXT NOT NULL,
                char_count INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                doc_name TEXT NOT NULL,
                section TEXT NOT NULL,
                page INTEGER,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                faiss_pos INTEGER NOT NULL UNIQUE
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
            """
        )
