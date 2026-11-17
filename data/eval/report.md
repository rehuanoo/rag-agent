# 企业知识库 RAG 评估报告

- 评估问题数：30
- 生成模型：deepseek-flash

## 检索指标

| 模式 | Hit@1 | Hit@3 | Hit@5 | MRR | Context Precision@5 |
| --- | --- | --- | --- | --- | --- |
| dense | 0.900 | 1.000 | 1.000 | 0.944 | 0.253 |
| hybrid | 1.000 | 1.000 | 1.000 | 1.000 | 0.273 |

## 回答质量（LLM 裁判）

| 指标 | 平均分 |
| --- | --- |
| Correctness | 1.0 |
| Faithfulness | 1.0 |
