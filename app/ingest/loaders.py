"""文档解析：Markdown / Word / PDF，保留标题层级与页码。"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
from docx import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

MARKDOWN_SUFFIXES = {".md", ".markdown"}
WORD_SUFFIXES = {".docx"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = MARKDOWN_SUFFIXES | WORD_SUFFIXES | PDF_SUFFIXES


@dataclass
class Section:
    """文档中的一个语义段：标题 + 正文，PDF 额外记录页码。"""

    title: str
    text: str
    page: int | None = None
    order: int = 0


@dataclass
class LoadedDoc:
    doc_id: str
    doc_name: str
    file_type: str
    source_path: str
    sections: list[Section] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(section.text) for section in self.sections)


def _make_doc_id(path: Path) -> str:
    return hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:10]


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_DOCX_HEADING = re.compile(r"^(?:Heading|标题)\s*([1-6])$", re.IGNORECASE)
_PDF_HEADING = re.compile(
    r"^\s*(?:"
    r"第[一二三四五六七八九十百零]+[章节部分篇]\s*.*"
    r"|[一二三四五六七八九十]+、\s*.+"
    r"|\d+(?:\.\d+){0,3}[、.\s]\s*.+"
    r")\s*$"
)
# 只匹配明确的页码标记，避免把表格里的纯数字单元格当成页码删掉
_PAGE_NOISE = re.compile(
    r"^\s*(?:第\s*\d+\s*页|Page\s*\d+.*|\d+\s*/\s*\d+)\s*$",
    re.IGNORECASE,
)
_PDF_HEADING_MIN_SIZE = 11.5


def _clean_text(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _looks_like_pdf_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 32:
        return False
    if stripped.endswith(("。", "；", "：", "，", "！", "？")):
        return False
    return bool(_PDF_HEADING.match(stripped))


def _flush_section(
    sections: list[Section],
    title: str,
    buffer: list[str],
    page: int | None,
) -> None:
    body = _clean_text("\n".join(buffer))
    if body:
        sections.append(Section(title=title or "正文", text=body, page=page, order=len(sections)))


def _heading_path(stack: list[tuple[int, str]]) -> str:
    """一级标题通常是文档名，引用时只保留更细的章节路径。"""
    if not stack:
        return "正文"
    titles = [title for _, title in stack]
    return " / ".join(titles[1:]) if len(titles) > 1 else titles[0]


def _load_markdown(path: Path, doc_id: str) -> LoadedDoc:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    sections: list[Section] = []
    heading_stack: list[tuple[int, str]] = []
    current_title = "概述"
    buffer: list[str] = []

    for line in raw.splitlines():
        match = _MD_HEADING.match(line)
        if match:
            _flush_section(sections, current_title, buffer, None)
            buffer = []

            level = len(match.group(1))
            title = match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_title = _heading_path(heading_stack)
        else:
            buffer.append(line)

    _flush_section(sections, current_title, buffer, None)
    return LoadedDoc(doc_id, path.stem, "markdown", str(path), sections)


def _iter_docx_blocks(document):
    """按文档真实顺序遍历段落和表格。"""
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def _load_docx(path: Path, doc_id: str) -> LoadedDoc:
    document = DocxDocument(str(path))
    sections: list[Section] = []
    heading_stack: list[tuple[int, str]] = []
    current_title = "概述"
    buffer: list[str] = []

    for block in _iter_docx_blocks(document):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            match = _DOCX_HEADING.match(block.style.name or "")
            if match:
                _flush_section(sections, current_title, buffer, None)
                buffer = []

                level = int(match.group(1))
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, text))
                current_title = _heading_path(heading_stack)
            else:
                buffer.append(text)
        else:
            rows = []
            for row in block.rows:
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                buffer.append("\n".join(rows))

    _flush_section(sections, current_title, buffer, None)
    return LoadedDoc(doc_id, path.stem, "word", str(path), sections)


def _load_pdf(path: Path, doc_id: str) -> LoadedDoc:
    """优先按字号识别标题（比正则可可靠），表格单元格字号小，不会被误判。"""
    sections: list[Section] = []
    current_title = "正文"
    buffer: list[str] = []

    with pymupdf.open(path) as pdf:
        for page_index in range(pdf.page_count):
            page_no = page_index + 1
            page_dict = pdf.load_page(page_index).get_text("dict")

            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    text = "".join(span.get("text", "") for span in spans).strip()
                    if not text or _PAGE_NOISE.match(text):
                        continue

                    font_size = max((span.get("size", 0) for span in spans), default=0)
                    if font_size >= _PDF_HEADING_MIN_SIZE:
                        _flush_section(sections, current_title, buffer, page_no)
                        buffer = []
                        current_title = text
                    else:
                        buffer.append(text)

            # 每一页结束时收口，保证章节不会跨页丢失页码信息
            _flush_section(sections, current_title, buffer, page_no)
            buffer = []
            current_title = "正文"

    # 字号信息缺失的 PDF 退化为文本启发式，避免整篇只剩一个“正文”
    if len(sections) <= 1:
        return _load_pdf_by_text(path, doc_id)

    return LoadedDoc(doc_id, path.stem, "pdf", str(path), sections)


def _load_pdf_by_text(path: Path, doc_id: str) -> LoadedDoc:
    sections: list[Section] = []
    current_title = "正文"
    buffer: list[str] = []

    with pymupdf.open(path) as pdf:
        for page_index in range(pdf.page_count):
            page_no = page_index + 1
            page_text = pdf.load_page(page_index).get_text("text")

            for line in page_text.splitlines():
                if _PAGE_NOISE.match(line):
                    continue
                if _looks_like_pdf_heading(line):
                    _flush_section(sections, current_title, buffer, page_no)
                    buffer = []
                    current_title = line.strip()
                else:
                    buffer.append(line)

            _flush_section(sections, current_title, buffer, page_no)
            buffer = []
            current_title = "正文"

    return LoadedDoc(doc_id, path.stem, "pdf", str(path), sections)


def load_document(path: Path) -> LoadedDoc:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支持的文件类型：{path.name}")

    doc_id = _make_doc_id(path)
    if suffix in MARKDOWN_SUFFIXES:
        return _load_markdown(path, doc_id)
    if suffix in WORD_SUFFIXES:
        return _load_docx(path, doc_id)
    return _load_pdf(path, doc_id)


def iter_raw_documents(raw_dir: Path) -> list[Path]:
    return sorted(
        path for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
