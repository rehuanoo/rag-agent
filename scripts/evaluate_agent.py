"""运行 Agent 路由与工具评估：python scripts/evaluate_agent.py [--limit N]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.eval.agent_eval import run
from app.logging_utils import setup_logging


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Agent 路由与工具评估")
    parser.add_argument("--limit", type=int, default=None, help="只评估前 N 条用例")
    args = parser.parse_args()
    run(limit=args.limit)


if __name__ == "__main__":
    main()
