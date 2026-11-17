"""Agent 路由与工具调用评估。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.agent.runner import AgentRunner
from app.config import PROJECT_ROOT
from app.eval.agent_cases import AGENT_CASES


@dataclass
class AgentCaseResult:
    question: str
    expected_route: str
    actual_route: str
    expected_tools: list[str]
    actual_tools: list[str]
    route_ok: bool
    tool_exact: bool
    tool_recall: float
    tool_precision: float
    latency: float
    answer: str


def _tool_metrics(expected: set[str], actual: set[str]) -> tuple[bool, float, float]:
    exact = expected == actual
    recall = len(expected & actual) / len(expected) if expected else 1.0
    if actual:
        precision = len(expected & actual) / len(actual)
    else:
        precision = 1.0 if not expected else 0.0
    return exact, recall, precision


def run(limit: int | None = None, out_dir: Path | None = None) -> dict:
    cases = AGENT_CASES[:limit] if limit else AGENT_CASES
    out_dir = out_dir or (PROJECT_ROOT / "data" / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    agent = AgentRunner()
    records: list[AgentCaseResult] = []

    for index, case in enumerate(cases, start=1):
        answer = agent.run(case["question"], thread_id=f"agent_eval_{index}")
        expected_tools = set(case["expected_tools"])
        actual_tools = {
            item.get("name", "")
            for item in answer.tool_calls
            if item.get("type") == "call"
        }
        exact, recall, precision = _tool_metrics(expected_tools, actual_tools)
        records.append(
            AgentCaseResult(
                question=case["question"],
                expected_route=case["expected_route"],
                actual_route=answer.route,
                expected_tools=sorted(expected_tools),
                actual_tools=sorted(actual_tools),
                route_ok=case["expected_route"] == answer.route,
                tool_exact=exact,
                tool_recall=recall,
                tool_precision=precision,
                latency=answer.elapsed,
                answer=answer.answer,
            )
        )
        print(
            f"[{index}/{len(cases)}] {case['question']} -> "
            f"route={answer.route} tools={sorted(actual_tools)} "
            f"route_ok={case['expected_route'] == answer.route}"
        )

    count = len(records) or 1
    summary = {
        "cases": len(records),
        "route_accuracy": sum(record.route_ok for record in records) / count,
        "tool_exact_match": sum(record.tool_exact for record in records) / count,
        "tool_recall": sum(record.tool_recall for record in records) / count,
        "tool_precision": sum(record.tool_precision for record in records) / count,
        "avg_latency": sum(record.latency for record in records) / count,
    }

    report = {
        "summary": summary,
        "details": [asdict(record) for record in records],
    }
    (out_dir / "agent_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Agent 路由与工具评估",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 用例数 | {summary['cases']} |",
        f"| 路由准确率 | {summary['route_accuracy']:.3f} |",
        f"| 工具完全匹配率 | {summary['tool_exact_match']:.3f} |",
        f"| 工具召回率 | {summary['tool_recall']:.3f} |",
        f"| 工具精确率 | {summary['tool_precision']:.3f} |",
        f"| 平均耗时 | {summary['avg_latency']:.2f}s |",
        "",
        "## 明细",
        "",
        "| 问题 | 期望路由 | 实际路由 | 期望工具 | 实际工具 | 路由正确 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        lines.append(
            f"| {record.question} | {record.expected_route} | {record.actual_route} | "
            f"{', '.join(record.expected_tools) or '-'} | "
            f"{', '.join(record.actual_tools) or '-'} | "
            f"{'是' if record.route_ok else '否'} |"
        )
    (out_dir / "agent_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"路由准确率      {summary['route_accuracy']:.3f}")
    print(f"工具完全匹配率  {summary['tool_exact_match']:.3f}")
    print(f"工具召回/精确   {summary['tool_recall']:.3f} / {summary['tool_precision']:.3f}")
    print(f"平均耗时        {summary['avg_latency']:.2f}s")
    print(f"报告：{out_dir / 'agent_report.md'}")
    return report
