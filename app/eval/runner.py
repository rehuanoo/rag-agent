"""RAG 评估：检索指标 + LLM 裁判回答质量。"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import LLM_PRIMARY_MODEL, PROJECT_ROOT
from app.eval.qa_set import QA_SET
from app.llm import create_chat_model
from app.rag.pipeline import RagPipeline
from app.retrieval.hybrid import RetrievalHit

JUDGE_SYSTEM_PROMPT = """你是严格的 RAG 评估员。根据问题、参考答案、系统回答和检索资料，给出两个 0 到 1 的分数：

1. correctness：系统回答与参考答案的关键事实是否一致。0 表示完全错误，1 表示关键事实完全正确，措辞不同不扣分。
2. faithfulness：系统回答中的事实是否都能在检索资料中找到依据。0 表示存在编造，1 表示全部有依据。

只输出 JSON，不要输出其他内容：
{"correctness": 0.0, "faithfulness": 0.0, "reason": "一句话说明"}"""


@dataclass
class RetrievalRecord:
    question: str
    hit1: bool
    hit3: bool
    hit5: bool
    mrr: float
    precision: float
    retrieved: list[str]


@dataclass
class AnswerRecord:
    question: str
    correctness: float | None
    faithfulness: float | None
    reason: str


def _is_relevant(hit: RetrievalHit, relevant: list[tuple[str, str]]) -> bool:
    return any(
        hit.doc_name == doc_name and hit.section == section
        for doc_name, section in relevant
    )


def _retrieval_record(question: str, hits: list[RetrievalHit], relevant) -> RetrievalRecord:
    flags = [_is_relevant(hit, relevant) for hit in hits]
    first = next((index for index, flag in enumerate(flags, start=1) if flag), None)
    return RetrievalRecord(
        question=question,
        hit1=bool(flags and flags[0]),
        hit3=any(flags[:3]),
        hit5=any(flags[:5]),
        mrr=1.0 / first if first else 0.0,
        precision=sum(flags) / len(flags) if flags else 0.0,
        retrieved=[f"{hit.doc_name}/{hit.section}" for hit in hits],
    )


def evaluate_retrieval(pipeline: RagPipeline, qa_set: list[dict], mode: str, use_rerank: bool) -> list[RetrievalRecord]:
    records = []
    for index, item in enumerate(qa_set, start=1):
        result = pipeline.retriever.search(
            item["question"],
            top_k=5,
            use_rerank=use_rerank,
            mode=mode,
        )
        records.append(_retrieval_record(item["question"], result.hits, item["relevant"]))
        print(f"  [{index}/{len(qa_set)}] {item['question']}")
    return records


def _summarize_retrieval(records: list[RetrievalRecord]) -> dict:
    count = len(records) or 1
    return {
        "questions": len(records),
        "hit@1": sum(record.hit1 for record in records) / count,
        "hit@3": sum(record.hit3 for record in records) / count,
        "hit@5": sum(record.hit5 for record in records) / count,
        "mrr": sum(record.mrr for record in records) / count,
        "precision@5": sum(record.precision for record in records) / count,
    }


def _judge(question: str, ground_truth: str, answer: str, contexts: list[str]) -> dict:
    context_text = "\n\n".join(contexts[:5])[:3000]
    prompt = (
        f"【问题】\n{question}\n\n"
        f"【参考答案】\n{ground_truth}\n\n"
        f"【系统回答】\n{answer[:1200]}\n\n"
        f"【检索资料】\n{context_text}"
    )
    llm = create_chat_model(temperature=0.0, max_tokens=300)

    for _ in range(2):
        try:
            response = llm.invoke(
                [SystemMessage(content=JUDGE_SYSTEM_PROMPT), HumanMessage(content=prompt)]
            )
            match = re.search(r"\{[\s\S]*\}", str(response.content))
            if not match:
                continue
            data = json.loads(match.group(0))
            return {
                "correctness": min(1.0, max(0.0, float(data.get("correctness", 0.0)))),
                "faithfulness": min(1.0, max(0.0, float(data.get("faithfulness", 0.0)))),
                "reason": str(data.get("reason", "")),
            }
        except Exception:  # noqa: BLE001 - 裁判失败时继续重试一次
            continue
    return {"correctness": None, "faithfulness": None, "reason": "裁判解析失败"}


def evaluate_answers(pipeline: RagPipeline, qa_set: list[dict]) -> list[AnswerRecord]:
    records = []
    for index, item in enumerate(qa_set, start=1):
        result = pipeline.answer(item["question"], use_rewrite=False)
        scores = _judge(
            item["question"],
            item["ground_truth"],
            result.answer,
            [hit.content for hit in result.hits],
        )
        records.append(
            AnswerRecord(
                question=item["question"],
                correctness=scores["correctness"],
                faithfulness=scores["faithfulness"],
                reason=scores["reason"],
            )
        )
        print(f"  [{index}/{len(qa_set)}] {item['question']} -> {scores['correctness']}")
    return records


def _summarize_answers(records: list[AnswerRecord]) -> dict:
    correctness = [record.correctness for record in records if record.correctness is not None]
    faithfulness = [record.faithfulness for record in records if record.faithfulness is not None]
    return {
        "questions": len(records),
        "correctness": sum(correctness) / len(correctness) if correctness else None,
        "faithfulness": sum(faithfulness) / len(faithfulness) if faithfulness else None,
        "judged": len(correctness),
    }


def _write_markdown(path: Path, retrieval: dict, answers: dict, summary: dict) -> None:
    lines = [
        "# 企业知识库 RAG 评估报告",
        "",
        f"- 评估问题数：{summary['question_count']}",
        f"- 生成模型：{summary['model']}",
        "",
        "## 检索指标",
        "",
        "| 模式 | Hit@1 | Hit@3 | Hit@5 | MRR | Context Precision@5 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for mode, data in retrieval.items():
        lines.append(
            f"| {mode} | {data['hit@1']:.3f} | {data['hit@3']:.3f} | "
            f"{data['hit@5']:.3f} | {data['mrr']:.3f} | {data['precision@5']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 回答质量（LLM 裁判）",
            "",
            "| 指标 | 平均分 |",
            "| --- | --- |",
            f"| Correctness | {answers['correctness'] if answers['correctness'] is not None else 'N/A'} |",
            f"| Faithfulness | {answers['faithfulness'] if answers['faithfulness'] is not None else 'N/A'} |",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(limit: int | None = None, out_dir: Path | None = None) -> dict:
    qa_set = QA_SET[:limit] if limit else QA_SET
    out_dir = out_dir or (PROJECT_ROOT / "data" / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = RagPipeline()
    retrieval_records: dict[str, list[RetrievalRecord]] = {}

    for mode, use_rerank in [("dense", False), ("hybrid", True)]:
        print(f"\n检索评估：{mode}")
        retrieval_records[mode] = evaluate_retrieval(pipeline, qa_set, mode, use_rerank)

    print("\n回答质量评估：hybrid + rerank")
    answer_records = evaluate_answers(pipeline, qa_set)

    retrieval_summary = {
        mode: _summarize_retrieval(records) for mode, records in retrieval_records.items()
    }
    answer_summary = _summarize_answers(answer_records)

    report = {
        "summary": {
            "question_count": len(qa_set),
            "model": LLM_PRIMARY_MODEL,
        },
        "retrieval": retrieval_summary,
        "answers": answer_summary,
        "retrieval_details": {
            mode: [asdict(record) for record in records]
            for mode, records in retrieval_records.items()
        },
        "answer_details": [asdict(record) for record in answer_records],
    }

    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_markdown(out_dir / "report.md", retrieval_summary, answer_summary, report["summary"])

    print("\n" + "=" * 60)
    print("检索指标")
    for mode, data in retrieval_summary.items():
        print(
            f"  {mode:<7} Hit@1={data['hit@1']:.3f} Hit@3={data['hit@3']:.3f} "
            f"Hit@5={data['hit@5']:.3f} MRR={data['mrr']:.3f} "
            f"Precision@5={data['precision@5']:.3f}"
        )
    print(
        f"回答质量  Correctness={answer_summary['correctness']}  "
        f"Faithfulness={answer_summary['faithfulness']}"
    )
    print(f"报告：{out_dir / 'report.md'}")
    return report
