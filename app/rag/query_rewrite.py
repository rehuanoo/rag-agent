"""Query 改写：用 LLM 生成 1-2 条补充查询，提升召回覆盖面。"""
from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import QUERY_REWRITE_MAX_QUERIES
from app.llm import chat_with_fallback

REWRITE_SYSTEM_PROMPT = """你是检索查询改写助手。根据用户问题生成 1 到 2 条更适合企业知识库检索的查询。

要求：
1. 保留原问题中的关键实体、数字、专有名词和制度名称；
2. 可以补充同义表达，或把一个多意图问题拆成两条；
3. 不要回答问题，不要解释；
4. 只输出 JSON 字符串数组，例如 ["查询一", "查询二"]。"""


def rewrite_query(question: str) -> list[str]:
    """返回去重后的查询列表，第一项始终是原始问题。失败时只返回原问题。"""
    try:
        text, _ = chat_with_fallback(
            [SystemMessage(content=REWRITE_SYSTEM_PROMPT), HumanMessage(content=question)]
        )
        match = re.search(r"\[[\s\S]*\]", text)
        items = json.loads(match.group(0)) if match else []
    except Exception:  # noqa: BLE001 - 改写失败不应阻断主流程
        items = []

    queries = [question]
    for item in items:
        item = str(item).strip()
        if item and item not in queries:
            queries.append(item)
    return queries[:QUERY_REWRITE_MAX_QUERIES]
