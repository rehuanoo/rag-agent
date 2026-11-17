"""把 corpus_data 中的语料生成 Markdown / Word / PDF 三种真实格式。"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docx import Document as DocxDocument
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.shared import Pt
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.config import RAW_DIR
from scripts.corpus_data import DOCS

CN_FONT = "STSong-Light"
PDF_TABLE_WIDTH = 515


def _parse_blocks(content: str):
    """把简化 Markdown 拆成标题、段落、列表和表格块。"""
    lines = content.splitlines()
    blocks = []
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            blocks.append(("heading", len(heading.group(1)), heading.group(2).strip()))
            index += 1
            continue

        if stripped.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            rows = []
            for row_index, row in enumerate(table_lines):
                cells = [cell.strip() for cell in row.strip("|").split("|")]
                if row_index == 1 and all(set(cell) <= {"-", ":", " "} for cell in cells):
                    continue
                rows.append(cells)
            if rows:
                blocks.append(("table", rows))
            continue

        if stripped.startswith("- "):
            blocks.append(("bullet", stripped[2:].strip()))
            index += 1
            continue

        blocks.append(("paragraph", stripped))
        index += 1

    return blocks


def _write_markdown(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _set_docx_run_font(run, size: float, bold: bool = False) -> None:
    run.font.name = "Microsoft YaHei"
    run.font.size = Pt(size)
    run.font.bold = bold
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")


def _write_docx(path: Path, title: str, content: str) -> None:
    document = DocxDocument()

    normal = document.styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")

    blocks = _parse_blocks(content)
    for block in blocks:
        kind = block[0]

        if kind == "heading":
            _, level, text = block
            if level == 1:
                paragraph = document.add_heading(level=1)
                run = paragraph.add_run(text)
                _set_docx_run_font(run, 18, bold=True)
            else:
                paragraph = document.add_heading(level=level)
                run = paragraph.add_run(text)
                _set_docx_run_font(run, 13 if level == 2 else 11.5, bold=True)
        elif kind == "paragraph":
            paragraph = document.add_paragraph()
            run = paragraph.add_run(block[1])
            _set_docx_run_font(run, 10.5)
        elif kind == "bullet":
            paragraph = document.add_paragraph(style="List Bullet")
            run = paragraph.add_run(block[1])
            _set_docx_run_font(run, 10.5)
        elif kind == "table":
            rows = block[1]
            column_count = max(len(row) for row in rows)
            table = document.add_table(rows=len(rows), cols=column_count)
            table.style = "Table Grid"
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            for row_index, row in enumerate(rows):
                for column_index in range(column_count):
                    cell = table.cell(row_index, column_index)
                    cell.text = row[column_index] if column_index < len(row) else ""
                    for paragraph in cell.paragraphs:
                        for run in paragraph.runs:
                            _set_docx_run_font(run, 9.5, bold=row_index == 0)

    document.save(str(path))


def _build_pdf_styles():
    return {
        "title": ParagraphStyle(
            "title", fontName=CN_FONT, fontSize=18, leading=24, spaceAfter=14
        ),
        "h2": ParagraphStyle(
            "h2", fontName=CN_FONT, fontSize=13, leading=19, spaceBefore=12, spaceAfter=6
        ),
        "h3": ParagraphStyle(
            "h3", fontName=CN_FONT, fontSize=11.5, leading=17, spaceBefore=8, spaceAfter=4
        ),
        "body": ParagraphStyle(
            "body", fontName=CN_FONT, fontSize=10.5, leading=17, spaceAfter=5
        ),
        "bullet": ParagraphStyle(
            "bullet", fontName=CN_FONT, fontSize=10.5, leading=17, leftIndent=14, spaceAfter=3
        ),
        "cell": ParagraphStyle("cell", fontName=CN_FONT, fontSize=8.8, leading=13),
    }


def _write_pdf(path: Path, title: str, content: str) -> None:
    pdfmetrics.registerFont(UnicodeCIDFont(CN_FONT))
    styles = _build_pdf_styles()

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont(CN_FONT, 9)
        canvas.drawRightString(555, 24, f"第 {doc.page} 页")
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=50,
        bottomMargin=45,
        title=title,
        author="星澜智能科技有限公司",
    )

    story = []
    for block in _parse_blocks(content):
        kind = block[0]
        if kind == "heading":
            _, level, text = block
            style = styles["title"] if level == 1 else styles["h2"] if level == 2 else styles["h3"]
            story.append(Paragraph(escape(text), style))
        elif kind == "paragraph":
            story.append(Paragraph(escape(block[1]), styles["body"]))
        elif kind == "bullet":
            story.append(Paragraph("• " + escape(block[1]), styles["bullet"]))
        elif kind == "table":
            rows = block[1]
            column_count = max(len(row) for row in rows)
            data = []
            for row in rows:
                data.append(
                    [
                        Paragraph(escape(row[i] if i < len(row) else ""), styles["cell"])
                        for i in range(column_count)
                    ]
                )
            table = Table(data, colWidths=[PDF_TABLE_WIDTH / column_count] * column_count)
            table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.5, (0.75, 0.75, 0.75)),
                        ("BACKGROUND", (0, 0), (-1, 0), (0.94, 0.95, 0.97)),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 8))

    document.build(story, onFirstPage=on_page, onLaterPages=on_page)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    counts = {"md": 0, "docx": 0, "pdf": 0}

    for item in DOCS:
        filename = item["filename"]
        content = item["content"]
        suffix = Path(filename).suffix.lower()
        path = RAW_DIR / filename
        title = filename.rsplit(".", 1)[0]

        if suffix == ".md":
            _write_markdown(path, content)
            counts["md"] += 1
        elif suffix == ".docx":
            _write_docx(path, title, content)
            counts["docx"] += 1
        elif suffix == ".pdf":
            _write_pdf(path, title, content)
            counts["pdf"] += 1
        else:
            raise ValueError(f"不支持的语料格式：{filename}")

        print(f"生成 {filename}")

    print(
        f"\n完成：Markdown {counts['md']} 篇，Word {counts['docx']} 篇，PDF {counts['pdf']} 篇"
        f"\n输出目录：{RAW_DIR}"
    )


if __name__ == "__main__":
    main()
