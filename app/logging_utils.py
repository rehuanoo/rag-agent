"""日志配置：控制台 + 轮转文件。"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from app.config import LOG_DIR


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if getattr(root, "_rag_configured", False):
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        LOG_DIR / "app.log",
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # 第三方 HTTP 客户端的 INFO 日志过于嘈杂
    for name in ("httpx", "httpx2", "httpcore", "urllib3", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)

    root._rag_configured = True  # type: ignore[attr-defined]
