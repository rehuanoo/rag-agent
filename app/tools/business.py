"""Agent 可调用的业务工具。"""
from __future__ import annotations

from langchain_core.tools import tool

from app.retrieval.hybrid import HybridRetriever

_RETRIEVER: HybridRetriever | None = None

CITY_STANDARDS = {
    "一线": {"hotel": 600, "allowance": 120},
    "二线": {"hotel": 450, "allowance": 100},
    "其他": {"hotel": 350, "allowance": 100},
}

EMPLOYEES = {
    "张伟": {
        "id": "E1024",
        "dept": "研发中心 / 后端组",
        "title": "高级后端工程师",
        "manager": "李娜",
        "office": "杭州总部 8F",
    },
    "李娜": {
        "id": "E1008",
        "dept": "研发中心 / 后端组",
        "title": "技术经理",
        "manager": "王强",
        "office": "杭州总部 8F",
    },
    "王强": {
        "id": "E1001",
        "dept": "研发中心",
        "title": "研发总监",
        "manager": "CTO 陈晨",
        "office": "杭州总部 9F",
    },
    "刘洋": {
        "id": "E2033",
        "dept": "客户成功部",
        "title": "客户成功经理",
        "manager": "周敏",
        "office": "深圳分部 12F",
    },
    "周敏": {
        "id": "E2001",
        "dept": "客户成功部",
        "title": "部门总监",
        "manager": "COO 赵磊",
        "office": "深圳分部 12F",
    },
    "陈晨": {
        "id": "E0002",
        "dept": "公司管理层",
        "title": "CTO",
        "manager": "CEO 林致远",
        "office": "杭州总部 10F",
    },
}

TOOL_AGENT_PROMPT = """你是星澜科技的业务助手，可以调用以下工具：
- knowledge_search：检索公司制度、产品、流程和常见问题
- travel_budget：按公司差旅标准计算住宿和补贴预算
- lookup_employee：查询员工工号、部门、职位和直属主管
- knowledge_base_stats：查询知识库文档和切片数量

规则：
1. 需要事实或数据时必须先调用工具，禁止凭记忆编造。
2. 工具返回“未找到”时如实说明，不要猜测。
3. 回答简洁，引用知识库时保留《文档》/章节 来源。"""


def configure_retriever(retriever: HybridRetriever) -> None:
    global _RETRIEVER
    _RETRIEVER = retriever


def _get_retriever() -> HybridRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = HybridRetriever()
    return _RETRIEVER


@tool
def knowledge_search(query: str) -> str:
    """在企业知识库中检索公司制度、产品、流程和常见问题。query 应为具体问题或关键词。"""
    result = _get_retriever().search(query, top_k=3, use_rerank=True)
    if not result.hits:
        return "知识库中没有检索到相关内容。"

    blocks = []
    for index, hit in enumerate(result.hits, start=1):
        page = f"，第 {hit.page} 页" if hit.page else ""
        blocks.append(f"[{index}] 《{hit.doc_name}》/ {hit.section}{page}\n{hit.content}")
    return "\n\n".join(blocks)


@tool
def travel_budget(city_tier: str, days: int) -> str:
    """按公司差旅标准计算预算。city_tier 只能是 一线、二线、其他；days 为出差天数。"""
    tier = city_tier.strip()
    if tier not in CITY_STANDARDS:
        return "城市级别无效，只能是：一线、二线、其他。"
    if days <= 0 or days > 60:
        return "出差天数需在 1 到 60 天之间。"

    standard = CITY_STANDARDS[tier]
    nights = max(days - 1, 1)
    hotel_total = standard["hotel"] * nights
    allowance_total = standard["allowance"] * days
    return (
        f"{tier}城市出差 {days} 天预算：\n"
        f"- 住宿：{standard['hotel']} 元/晚 × {nights} 晚 = {hotel_total} 元\n"
        f"- 补贴：{standard['allowance']} 元/天 × {days} 天 = {allowance_total} 元\n"
        f"- 合计：{hotel_total + allowance_total} 元\n"
        f"说明：住宿按 {nights} 晚估算（出差天数减 1），超标部分需个人承担。"
    )


@tool
def lookup_employee(name: str) -> str:
    """按姓名或工号查询员工信息，包括部门、职位、直属主管和办公地点。"""
    key = name.strip()
    if not key:
        return "请提供员工姓名或工号。"

    matches = []
    for employee_name, info in EMPLOYEES.items():
        if key == employee_name or key.upper() == info["id"] or key in employee_name:
            matches.append((employee_name, info))

    if not matches:
        return f"员工目录中未找到“{name}”。当前目录仅包含部分示例员工。"
    if len(matches) > 1:
        return "匹配到多位员工：" + "、".join(item[0] for item in matches) + "，请提供更完整的姓名。"

    employee_name, info = matches[0]
    return (
        f"{employee_name}（工号 {info['id']}）\n"
        f"- 部门：{info['dept']}\n"
        f"- 职位：{info['title']}\n"
        f"- 直属主管：{info['manager']}\n"
        f"- 办公地点：{info['office']}"
    )


@tool
def knowledge_base_stats() -> str:
    """查询当前知识库的文档数量、切片数量和存储方式。"""
    store = _get_retriever().store
    docs, chunks = store.stats()
    return (
        f"当前知识库包含 {docs} 篇文档、{chunks} 个切片；"
        "向量模型 bge-large-zh-v1.5，向量库 FAISS，元数据 SQLite。"
    )


TOOLS = [knowledge_search, travel_budget, lookup_employee, knowledge_base_stats]
