"""LangGraph 工作流：路由 -> 检索 -> 评分 -> 改写/生成，工具走 create_agent。"""
from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from app.agent.state import AgentRuntime, AgentState
from app.config import AGENT_MAX_REWRITES
from app.rag.prompts import SYSTEM_PROMPT, build_user_prompt, format_context
from app.rag.query_rewrite import rewrite_query

ROUTER_PROMPT = """你是星澜科技企业助手的路由器。判断用户问题应该走哪条路径：

- knowledge：查询公司制度、产品、流程、政策、FAQ 等知识库内容
- tools：需要计算差旅预算、查询员工信息、查询知识库统计等业务工具
- direct：寒暄、能力介绍、与公司业务无关的通用问题

示例：
问：P0 工单多久响应？ -> knowledge
问：帮我算一下去上海出差 3 天的预算 -> tools
问：张伟是哪个部门的？ -> tools
问：知识库里有多少个切片？ -> tools
问：你好，你能做什么？ -> direct

只输出 JSON，不要解释：
{"route": "knowledge|tools|direct", "reason": "不超过 20 字"}"""

GRADER_PROMPT = """判断【资料】是否足以回答【问题】。

判定标准：
- 资料中包含回答问题所需的关键事实、数字或条件 -> sufficient=true
- 资料只是沾边、缺少关键数字或条件 -> sufficient=false

只输出 JSON，不要解释：
{"sufficient": true, "reason": "不超过 20 字"}"""

DIRECT_PROMPT = """你是星澜科技的企业知识库助手。你可以查询公司制度、产品、流程和 FAQ，
也可以计算差旅预算、查询员工信息和知识库统计。回答简洁友好，涉及具体政策时提醒用户可以继续追问。"""

CONTEXTUALIZE_PROMPT = """你是对话改写助手。根据历史对话，把用户最后一个问题改写成一个不依赖上下文、可独立检索或执行的问题。

规则：
1. 保留关键实体、数字和专有名词；
2. 补全“他/她/这个/上面说的”等指代；
3. 不要回答问题，不要解释；
4. 只输出改写后的问题。"""


def _contextualize_node(runtime: AgentRuntime):
    def node(state: AgentState) -> dict:
        original = str(state["messages"][-1].content)
        prior = state["messages"][:-1][-4:]
        if not prior:
            return {
                "original_question": original,
                "question": original,
                "steps": state.get("steps", []) + ["contextualize → 首轮，无需改写"],
            }

        history_lines = []
        for message in prior:
            role = "用户" if message.type == "human" else "助手"
            history_lines.append(f"{role}：{str(message.content)[:300]}")
        try:
            response = runtime.router_llm.invoke(
                [
                    SystemMessage(content=CONTEXTUALIZE_PROMPT),
                    HumanMessage(
                        content="历史对话：\n"
                        + "\n".join(history_lines)
                        + f"\n\n最后一个问题：{original}"
                    ),
                ]
            )
            standalone = str(response.content).strip()
        except Exception:  # noqa: BLE001 - 改写失败时使用原问题
            standalone = ""

        if not standalone or len(standalone) > 200:
            standalone = original
            note = "contextualize → 保持原问题"
        else:
            note = "contextualize → 已补全上下文"
        return {
            "original_question": original,
            "question": standalone,
            "steps": state.get("steps", []) + [note],
        }

    return node


def _json_call(llm, system_prompt: str, user_content: str, default: dict) -> dict:
    try:
        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
        )
        match = re.search(r"\{[\s\S]*\}", str(response.content))
        if match:
            return json.loads(match.group(0))
    except Exception:  # noqa: BLE001 - 解析失败时使用默认决策
        pass
    return default


def _router_node(runtime: AgentRuntime):
    def node(state: AgentState) -> dict:
        question = state["question"]
        decision = _json_call(
            runtime.router_llm,
            ROUTER_PROMPT,
            f"用户问题：{question}",
            {"route": "knowledge", "reason": "默认走知识库"},
        )
        route = str(decision.get("route", "knowledge")).strip()
        if route not in {"knowledge", "tools", "direct"}:
            route = "knowledge"
        return {
            "question": question,
            "route": route,
            "route_reason": str(decision.get("reason", "")),
            "steps": state.get("steps", []) + [f"router → {route}"],
        }

    return node


def _retrieve_node(runtime: AgentRuntime):
    def node(state: AgentState) -> dict:
        question = state["question"]
        queries = state.get("queries") or [question]
        result = runtime.retriever.search(
            question,
            top_k=5,
            use_rerank=True,
            extra_queries=queries[1:],
        )
        documents = [
            {
                "doc_name": hit.doc_name,
                "section": hit.section,
                "page": hit.page,
                "content": hit.content,
                "score": round(hit.score, 4),
            }
            for hit in result.hits
        ]
        return {
            "documents": documents,
            "steps": state.get("steps", [])
            + [f"retrieve → {len(documents)} 条候选（{result.elapsed:.2f}s）"],
        }

    return node


def _grade_node(runtime: AgentRuntime):
    def node(state: AgentState) -> dict:
        documents = state.get("documents", [])
        if not documents:
            return {
                "grade": "insufficient",
                "grade_reason": "没有检索到资料",
                "steps": state.get("steps", []) + ["grade → insufficient（无资料）"],
            }

        decision = _json_call(
            runtime.grader_llm,
            GRADER_PROMPT,
            f"【问题】\n{state['question']}\n\n【资料】\n{format_context(documents)[:3000]}",
            {"sufficient": True, "reason": "评分失败，默认通过"},
        )
        sufficient = bool(decision.get("sufficient", True))
        return {
            "grade": "sufficient" if sufficient else "insufficient",
            "grade_reason": str(decision.get("reason", "")),
            "steps": state.get("steps", [])
            + [f"grade → {'sufficient' if sufficient else 'insufficient'}"],
        }

    return node


def _rewrite_node():
    def node(state: AgentState) -> dict:
        queries = rewrite_query(state["question"])
        return {
            "queries": queries,
            "rewrite_count": state.get("rewrite_count", 0) + 1,
            "steps": state.get("steps", [])
            + [f"rewrite → 第 {state.get('rewrite_count', 0) + 1} 次，生成 {len(queries)} 条查询"],
        }

    return node


def _generate_node(runtime: AgentRuntime):
    def node(state: AgentState) -> dict:
        documents = state.get("documents", [])
        prior = state["messages"][:-1][-4:]
        prompt = build_user_prompt(state["question"], format_context(documents))
        response = runtime.llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), *prior, HumanMessage(content=prompt)]
        )
        answer = str(response.content).strip()
        citations = [
            {
                "doc_name": item["doc_name"],
                "section": item["section"],
                "page": item.get("page"),
                "score": item.get("score", 0.0),
            }
            for item in documents
        ]
        return {
            "answer": answer,
            "citations": citations,
            "messages": [AIMessage(content=answer)],
            "steps": state.get("steps", []) + ["generate → 完成"],
        }

    return node


def _insufficient_node():
    def node(state: AgentState) -> dict:
        reason = state.get("grade_reason") or "资料不足"
        answer = f"知识库中没有找到足够的依据来回答这个问题。\n\n原因：{reason}"
        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)],
            "steps": state.get("steps", []) + ["insufficient → 拒绝作答"],
        }

    return node


def _tools_node(runtime: AgentRuntime):
    def node(state: AgentState) -> dict:
        # 只传当前这一轮的问题，避免工具 Agent 被历史回答带偏
        try:
            result = runtime.tool_agent.invoke(
                {"messages": [HumanMessage(content=state["question"])]}
            )
            messages = result["messages"]
            answer = str(messages[-1].content).strip()

            trace: list[dict] = []
            for message in messages:
                for call in getattr(message, "tool_calls", None) or []:
                    trace.append(
                        {
                            "type": "call",
                            "name": call.get("name", ""),
                            "args": call.get("args", {}),
                        }
                    )
                if isinstance(message, ToolMessage):
                    trace.append(
                        {
                            "type": "result",
                            "name": getattr(message, "name", "tool"),
                            "content": str(message.content)[:400],
                        }
                    )
        except Exception as exc:  # noqa: BLE001 - 工具失败要反馈给用户
            answer = f"工具调用失败：{exc}"
            trace = []

        return {
            "answer": answer,
            "tool_calls": trace,
            "messages": [AIMessage(content=answer)],
            "steps": state.get("steps", [])
            + [f"tools → {sum(1 for item in trace if item['type'] == 'call')} 次调用"],
        }

    return node


def _direct_node(runtime: AgentRuntime):
    def node(state: AgentState) -> dict:
        prior = state["messages"][:-1][-4:]
        response = runtime.llm.invoke(
            [SystemMessage(content=DIRECT_PROMPT), *prior, HumanMessage(content=state["question"])]
        )
        answer = str(response.content).strip()
        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)],
            "steps": state.get("steps", []) + ["direct → 直接回答"],
        }

    return node


def _after_grade(state: AgentState) -> str:
    if state.get("grade") == "sufficient":
        return "generate"
    if state.get("rewrite_count", 0) >= AGENT_MAX_REWRITES:
        return "insufficient"
    return "rewrite"


def build_agent_graph(runtime: AgentRuntime):
    builder = StateGraph(AgentState)
    builder.add_node("contextualize", _contextualize_node(runtime))
    builder.add_node("router", _router_node(runtime))
    builder.add_node("retrieve", _retrieve_node(runtime))
    builder.add_node("grade", _grade_node(runtime))
    builder.add_node("rewrite", _rewrite_node())
    builder.add_node("generate", _generate_node(runtime))
    builder.add_node("insufficient", _insufficient_node())
    builder.add_node("tools", _tools_node(runtime))
    builder.add_node("direct", _direct_node(runtime))

    builder.add_edge(START, "contextualize")
    builder.add_edge("contextualize", "router")
    builder.add_conditional_edges(
        "router",
        lambda state: state.get("route", "knowledge"),
        {"knowledge": "retrieve", "tools": "tools", "direct": "direct"},
    )
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        _after_grade,
        {"generate": "generate", "rewrite": "rewrite", "insufficient": "insufficient"},
    )
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("generate", END)
    builder.add_edge("insufficient", END)
    builder.add_edge("tools", END)
    builder.add_edge("direct", END)

    return builder.compile(checkpointer=runtime.memory)
