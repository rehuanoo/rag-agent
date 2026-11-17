"""结构感知切分：先按章节，再按中文标点递归切块。"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.ingest.loaders import LoadedDoc


@dataclass
class Chunk:
    doc_id: str
    doc_name: str
    section: str
    page: int | None
    chunk_index: int
    content: str


def _build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""],
    )


def split_document(doc: LoadedDoc) -> list[Chunk]:
    splitter = _build_splitter()
    chunks: list[Chunk] = []

    for section in doc.sections:
        text = section.text.strip()
        if not text:
            continue

        # 标题写进 chunk，向量化后能保留“这段话属于哪个制度”的语义
        header = f"《{doc.doc_name}》· {section.title}"
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    doc_id=doc.doc_id,
                    doc_name=doc.doc_name,
                    section=section.title,
                    page=section.page,
                    chunk_index=len(chunks),
                    content=f"{header}\n{piece}",
                )
            )

    return chunks
