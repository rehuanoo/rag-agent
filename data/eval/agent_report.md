# Agent 路由与工具评估

| 指标 | 数值 |
| --- | --- |
| 用例数 | 12 |
| 路由准确率 | 1.000 |
| 工具完全匹配率 | 0.833 |
| 工具召回率 | 1.000 |
| 工具精确率 | 0.917 |
| 平均耗时 | 3.84s |

## 明细

| 问题 | 期望路由 | 实际路由 | 期望工具 | 实际工具 | 路由正确 |
| --- | --- | --- | --- | --- | --- |
| P0 工单多久响应？ | knowledge | knowledge | - | - | 是 |
| 年假有多少天？ | knowledge | knowledge | - | - | 是 |
| 报销发票抬头和税号是什么？ | knowledge | knowledge | - | - | 是 |
| 帮我算一下去北京出差 5 天的预算 | tools | tools | travel_budget | knowledge_search, travel_budget | 是 |
| 张伟的工号是多少？ | tools | tools | lookup_employee | lookup_employee | 是 |
| 知识库里有多少个切片？ | tools | tools | knowledge_base_stats | knowledge_base_stats | 是 |
| 你好，你能做什么？ | direct | direct | - | - | 是 |
| 深圳出差 2 天，住宿和补贴大概多少？ | tools | tools | travel_budget | knowledge_search, travel_budget | 是 |
| 私有化部署最低配置是什么？ | knowledge | knowledge | - | - | 是 |
| 周敏是哪个部门的？ | tools | tools | lookup_employee | lookup_employee | 是 |
| 谢谢，再见 | direct | direct | - | - | 是 |
| 公司有没有班车？ | knowledge | knowledge | - | - | 是 |
