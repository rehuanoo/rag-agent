"""Agent 评估用例：期望路由与期望工具。"""

AGENT_CASES = [
    {
        "question": "P0 工单多久响应？",
        "expected_route": "knowledge",
        "expected_tools": [],
    },
    {
        "question": "年假有多少天？",
        "expected_route": "knowledge",
        "expected_tools": [],
    },
    {
        "question": "报销发票抬头和税号是什么？",
        "expected_route": "knowledge",
        "expected_tools": [],
    },
    {
        "question": "帮我算一下去北京出差 5 天的预算",
        "expected_route": "tools",
        "expected_tools": ["travel_budget"],
    },
    {
        "question": "张伟的工号是多少？",
        "expected_route": "tools",
        "expected_tools": ["lookup_employee"],
    },
    {
        "question": "知识库里有多少个切片？",
        "expected_route": "tools",
        "expected_tools": ["knowledge_base_stats"],
    },
    {
        "question": "你好，你能做什么？",
        "expected_route": "direct",
        "expected_tools": [],
    },
    {
        "question": "深圳出差 2 天，住宿和补贴大概多少？",
        "expected_route": "tools",
        "expected_tools": ["travel_budget"],
    },
    {
        "question": "私有化部署最低配置是什么？",
        "expected_route": "knowledge",
        "expected_tools": [],
    },
    {
        "question": "周敏是哪个部门的？",
        "expected_route": "tools",
        "expected_tools": ["lookup_employee"],
    },
    {
        "question": "谢谢，再见",
        "expected_route": "direct",
        "expected_tools": [],
    },
    {
        "question": "公司有没有班车？",
        "expected_route": "knowledge",
        "expected_tools": [],
    },
]
