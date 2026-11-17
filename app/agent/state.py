"""LangGraph Agent 的状态与输出结构。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    original_question: str
    question: str
    route: str
    route_reason: str
    queries: list[str]
    documents: list[dict]
    grade: str
    grade_reason: str
    rewrite_count: int
    answer: str
    citations: list[dict]
    tool_calls: list[dict]
    steps: list[str]


@dataclass
class AgentAnswer:
    answer: str
    route: str
    route_reason: str
    original_question: str = ""
    question: str = ""
    steps: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    elapsed: float = 0.0


@dataclass
class AgentRuntime:
    """节点共享的运行时依赖，避免在状态里存不可序列化的对象。"""

    retriever: Any
    llm: Any
    router_llm: Any
    grader_llm: Any
    tool_agent: Any
    memory: Any
