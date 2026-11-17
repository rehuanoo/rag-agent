"""Agent 运行入口。"""
from __future__ import annotations

import logging
import time

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.builder import build_agent_graph
from app.agent.state import AgentAnswer, AgentRuntime
from app.llm import create_chat_model
from app.retrieval.hybrid import HybridRetriever
from app.tools.business import TOOL_AGENT_PROMPT, TOOLS, configure_retriever

logger = logging.getLogger(__name__)


class AgentRunner:
    def __init__(self, retriever: HybridRetriever | None = None) -> None:
        self.retriever = retriever or HybridRetriever()
        configure_retriever(self.retriever)

        self.runtime = AgentRuntime(
            retriever=self.retriever,
            llm=create_chat_model(),
            router_llm=create_chat_model(temperature=0.0, max_tokens=200),
            grader_llm=create_chat_model(temperature=0.0, max_tokens=200),
            tool_agent=create_agent(
                model=create_chat_model(),
                tools=TOOLS,
                system_prompt=TOOL_AGENT_PROMPT,
            ),
            memory=InMemorySaver(),
        )
        self.graph = build_agent_graph(self.runtime)

    def run(self, question: str, thread_id: str = "web") -> AgentAnswer:
        started = time.perf_counter()
        state = self.graph.invoke(
            {
                "messages": [HumanMessage(content=question)],
                "original_question": question,
                "question": question,
                "steps": [],
                "citations": [],
                "tool_calls": [],
                "queries": [],
                "documents": [],
                "rewrite_count": 0,
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        agent_answer = AgentAnswer(
            answer=state.get("answer", ""),
            route=state.get("route", ""),
            route_reason=state.get("route_reason", ""),
            original_question=state.get("original_question", question),
            question=state.get("question", question),
            steps=state.get("steps", []),
            citations=state.get("citations", []),
            tool_calls=state.get("tool_calls", []),
            elapsed=time.perf_counter() - started,
        )
        logger.info(
            "agent run thread=%s route=%s tools=%d steps=%d elapsed=%.2fs question=%r",
            thread_id,
            agent_answer.route,
            sum(1 for item in agent_answer.tool_calls if item.get("type") == "call"),
            len(agent_answer.steps),
            agent_answer.elapsed,
            question[:80],
        )
        return agent_answer
