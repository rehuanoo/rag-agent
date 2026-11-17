"""命令行文档管理。

用法：
    python scripts/manage_docs.py list
    python scripts/manage_docs.py add "C:\\path\\doc.pdf"
    python scripts/manage_docs.py remove 员工手册
    python scripts/manage_docs.py rebuild
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingest.pipeline import IngestionService
from app.logging_utils import setup_logging


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="企业知识库文档管理")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="列出已入库文档")
    add_parser = sub.add_parser("add", help="新增或更新文档")
    add_parser.add_argument("path", help="文档路径")
    add_parser.add_argument(
        "--no-copy",
        action="store_true",
        help="不复制到 data/raw，直接使用原路径",
    )
    remove_parser = sub.add_parser("remove", help="删除文档并重建索引")
    remove_parser.add_argument("name", help="文档名（不含扩展名）")
    sub.add_parser("rebuild", help="扫描 data/raw 全量重建索引")

    args = parser.parse_args()
    service = IngestionService()

    if args.command == "list":
        rows = service.store.list_documents()
        if not rows:
            print("知识库为空")
            return
        print(f"{'文档名':<28}{'类型':<10}{'切片':>6}{'字符数':>10}")
        print("-" * 56)
        for row in rows:
            print(
                f"{row['doc_name']:<28}{row['file_type']:<10}"
                f"{row['chunk_count']:>6}{row['char_count']:>10}"
            )
        return

    if args.command == "add":
        result = service.add_file(Path(args.path), copy_to_raw=not args.no_copy)
        print(
            f"已入库《{result.doc_name}》：{result.chunks} 个切片，"
            f"{result.chars} 字符，类型 {result.file_type}"
        )
        return

    if args.command == "remove":
        info = service.remove_document(args.name)
        print(
            f"已删除《{info['doc_name']}》，移除 {info['deleted_chunks']} 个切片；"
            f"剩余 {info['remaining_docs']} 篇文档、{info['remaining_chunks']} 个切片"
        )
        return

    info = service.rebuild()
    print(f"重建完成：{info['docs']} 篇文档，{info['chunks']} 个切片")


if __name__ == "__main__":
    main()
