"""RAG 共用的系统提示词与上下文格式化。"""
from __future__ import annotations

SYSTEM_PROMPT = """你是星澜智能科技的企业知识库助手，回答必须严格依据提供的参考资料。

规则：
1. 只使用【参考资料】中的信息，禁止编造、推测或补充资料外的内容。
2. 资料不足以回答时，明确说明“知识库中没有找到相关依据”，并指出缺少哪类信息。
3. 涉及金额、天数、时限、比例、审批人时，直接引用原文中的具体数值，不要模糊表达。
4. 每条结论后用 [编号] 标注引用来源，例如 [1][2]。
5. 用中文回答，先给结论，再给必要说明，不要大段复述资料。
"""


def format_context(items) -> str:
    """兼容 RetrievalHit 和 dict 两种候选格式。"""
    blocks = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            doc_name = item["doc_name"]
            section = item["section"]
            page = item.get("page")
            content = item["content"]
        else:
            doc_name = item.doc_name
            section = item.section
            page = item.page
            content = item.content

        source = f"《{doc_name}》｜{section}"
        if page:
            source += f"｜第 {page} 页"
        blocks.append(f"[{index}] 来源：{source}\n{content}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, context: str) -> str:
    return (
        f"【参考资料】\n{context}\n\n"
        f"【问题】\n{question}\n\n"
        "请依据参考资料作答，并在结论后标注引用编号。"
    )
