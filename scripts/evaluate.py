"""运行 RAG 评估：python scripts/evaluate.py [--limit N]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.eval.runner import run
from app.logging_utils import setup_logging


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="企业知识库 RAG 评估")
    parser.add_argument("--limit", type=int, default=None, help="只评估前 N 条问题")
    args = parser.parse_args()
    run(limit=args.limit)


if __name__ == "__main__":
    main()
