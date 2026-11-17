# 星澜科技企业知识库 RAG Agent

一个用于学习 **RAG、LangChain、LangGraph、Tool Calling** 的完整项目：从企业文档解析、混合检索、精排，到带引用的问答、检索调试、LangGraph Agent 和自动化评估。

## 功能

- **文档入库**：Markdown / PDF / Word 解析，结构感知切分，增量新增、删除、全量重建
- **混合检索**：bge-large-zh 向量召回 + jieba/BM25 关键词召回 → RRF 融合 → bge-reranker-base 精排
- **带引用问答**：回答标注 `[编号]`，界面可回溯到文档、章节和页码
- **检索调试**：逐条查看向量排名/分数、BM25 排名/分数、RRF、精排分数
- **LangGraph Agent**：上下文改写 → 路由 → 检索 → 资料评分 → 改写重试 / 生成；工具问题交给 `create_agent`
- **工具调用**：知识库检索、差旅预算计算、员工查询、知识库统计
- **多轮记忆**：LangGraph Checkpointer + 指代消解（“他的主管” → “张伟的主管”）
- **评估**：30 条 RAG 测试集 + 12 条 Agent 路由/工具测试集
- **工程化**：日志轮转、单并发保护、文档管理页

## 架构

```
                         ┌──────────────────────────────┐
用户问题 ──► 上下文改写 ──►│            Router            │
                         └──────┬───────────┬───────────┘
                                │           │
                    knowledge   │           │ tools
                                ▼           ▼
                         ┌──────────┐   ┌─────────────────────┐
                         │ Retrieve │   │ create_agent        │
                         │ 向量+BM25 │   │ knowledge_search    │
                         │ → RRF    │   │ travel_budget       │
                         │ → Rerank │   │ lookup_employee     │
                         └────┬─────┘   │ knowledge_base_stats│
                              ▼         └──────────┬──────────┘
                         ┌──────────┐              │
                         │  Grade   │              │
                         └──┬────┬──┘              │
                   充分     │    │ 不充分            │
                            ▼    ▼                  │
                        Generate  Rewrite（最多 2 次）│
                            │        │              │
                            └────────┴──────────────┘
                                     ▼
                                  回答 + 引用
```

## 目录

```
rag-enterprise-agent/
├── app/
│   ├── config.py            # 路径、模型、检索与 Agent 参数
│   ├── llm.py               # DeepSeek 客户端 + 多模型降级
│   ├── local_llm.py         # 本地 Qwen 兜底（默认关闭）
│   ├── embeddings.py        # bge-large-zh + mean pooling
│   ├── logging_utils.py     # 控制台 + 轮转文件日志
│   ├── ingest/
│   │   ├── loaders.py       # Markdown / Word / PDF 解析
│   │   ├── splitter.py      # 结构感知切分
│   │   └── pipeline.py      # 增量新增 / 删除 / 重建
│   ├── store/faiss_store.py # FAISS + SQLite 元数据
│   ├── retrieval/           # dense / bm25 / rrf / reranker / hybrid
│   ├── rag/                 # prompts / query_rewrite / pipeline
│   ├── tools/business.py    # 四个业务工具
│   ├── agent/               # state / builder / runner
│   ├── eval/                # RAG 与 Agent 评估
│   └── ui.py                # Gradio：对话 / 调试 / Agent / 文档管理
├── data/
│   ├── raw/                 # 原始文档
│   ├── index/               # faiss.index + metadata.db
│   ├── eval/                # 评估报告
│   └── traces/              # 预留轨迹目录
├── logs/app.log             # 运行日志
├── scripts/
│   ├── corpus_data.py       # 15 篇模拟企业语料
│   ├── generate_docs.py     # 生成 md / docx / pdf
│   ├── build_index.py       # 全量建索引
│   ├── manage_docs.py       # list / add / remove / rebuild
│   ├── evaluate.py          # RAG 评估
│   └── evaluate_agent.py    # Agent 评估
└── requirements.txt
```

## 快速开始

### 1. 环境

本项目复用已有虚拟环境：

```powershell
S:\PythonAILearningCourse\6.Transformers\.venv\Scripts\python.exe
```

依赖安装：

```powershell
& "S:\PythonAILearningCourse\6.Transformers\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

### 2. 模型

| 模型 | 路径 |
|---|---|
| bge-large-zh-v1.5 | `S:\PythonAILearningCourse\6.Transformers\models\bge-large-zh-v1.5` |
| bge-reranker-base | `S:\PythonAILearningCourse\6.Transformers\models\bge-reranker-base` |

如需重新下载精排模型：

```powershell
modelscope download --model BAAI/bge-reranker-base --local_dir "S:\PythonAILearningCourse\6.Transformers\models\bge-reranker-base"
```

### 3. API Key

使用 DeepSeek，Key 读取顺序：

1. 进程环境变量 `dsAPIkey`
2. Windows 系统级环境变量 `dsAPIkey`
3. 进程环境变量 `DEEPSEEK_API_KEY`

```powershell
# 当前会话临时设置
$env:dsAPIkey = "your-key"
```

### 4. 生成语料并建索引

```powershell
cd "P:\My ai prog\rag-enterprise-agent"
& "S:\PythonAILearningCourse\6.Transformers\.venv\Scripts\python.exe" scripts\generate_docs.py
& "S:\PythonAILearningCourse\6.Transformers\.venv\Scripts\python.exe" scripts\build_index.py
```

### 5. 启动界面

```powershell
& "S:\PythonAILearningCourse\6.Transformers\.venv\Scripts\python.exe" app\ui.py
```

也可以直接双击项目根目录的 `start.bat`：会自动切换目录、检查虚拟环境、缺少索引时自动构建，然后启动服务并打开浏览器。

打开 http://127.0.0.1:7860，包含四个页签：

- **对话**：混合检索 + 引用
- **检索调试**：四层得分
- **Agent**：路由、执行轨迹、工具调用
- **文档管理**：上传、删除、重建

## 文档管理

```powershell
# 列出文档
python scripts/manage_docs.py list

# 新增或更新（会复制到 data/raw）
python scripts/manage_docs.py add "C:\docs\新制度.pdf"

# 删除文档并重建剩余索引
python scripts/manage_docs.py remove 新制度

# 扫描 data/raw 全量重建
python scripts/manage_docs.py rebuild
```

## 评估

```powershell
# RAG：30 条问题，对比纯向量 vs 混合检索
python scripts/evaluate.py

# Agent：12 条路由与工具用例
python scripts/evaluate_agent.py
```

当前结果：

**RAG 检索**

| 模式 | Hit@1 | Hit@3 | Hit@5 | MRR | Precision@5 |
|---|---|---|---|---|---|
| 纯向量 | 0.900 | 1.000 | 1.000 | 0.944 | 0.253 |
| 混合 + 精排 | **1.000** | 1.000 | 1.000 | **1.000** | 0.273 |

**RAG 回答**：Correctness 1.000，Faithfulness 1.000（LLM 裁判）。

**Agent**

| 指标 | 数值 |
|---|---|
| 路由准确率 | 1.000 |
| 工具完全匹配率 | 0.833 |
| 工具召回率 | 1.000 |
| 工具精确率 | 0.917 |
| 平均耗时 | 3.84s |

工具完全匹配率较低，是因为差旅预算问题会额外调用 `knowledge_search` 核对公司标准，属于有意的二次校验，不是错误调用。

报告位置：

- `data/eval/report.md` / `report.json`
- `data/eval/agent_report.md` / `agent_report.json`

## 关键设计取舍

### 1. 为什么不用 sentence-transformers 加载 bge

本地 bge 模型的 `modules.json` 是旧格式，`sentence-transformers 6.0.1` 会报 `Pooling.__init__() missing embedding_dimension`。因此改为 `transformers.AutoModel + mean pooling`，同时按 bge 官方建议给 query 加指令前缀：

```
为这个句子生成表示以用于检索相关文章：
```

### 2. 为什么 FAISS + SQLite

FAISS 只负责向量和相似度，不支持元数据过滤、文档删除和列表查询。因此：

- 向量：`faiss.IndexFlatIP`，向量已 L2 归一化，内积等价余弦相似度
- 元数据：SQLite 保存文档、章节、页码、切片和 `faiss_pos` 映射
- 新增：直接追加向量，不重建已有索引
- 删除：删除切片后对剩余切片重新向量化并重排 `faiss_pos`

### 3. PDF 解析

- 标题按**字号**识别（正文 10.5pt，标题 12pt 以上），比正则可靠
- 表格单元格是独立文本行，不能把纯数字当页码过滤，否则会丢掉“并发 2000”这类关键数据
- 页码过滤只匹配 `第 X 页`、`Page X`、`X/Y`

### 4. 混合检索与精排

- BM25 用 `jieba.lcut_for_search` 分词，适合中文
- RRF 只使用名次，避免余弦分数和 BM25 分数不可比
- 精排取 RRF 前 20 条，Cross-Encoder 逐对打分后取 top5

### 5. DeepSeek 的限制

DeepSeek 的 thinking 模式不支持强制 `tool_choice`，因此：

- 路由、资料评分：用 JSON 提示词 + 正则解析，不用 `with_structured_output`
- 工具调用：`create_agent` 的自动 tool calling 正常

### 6. 为什么没有直接装 RAGAS

`ragas` 会把共享 venv 中的 `langchain-openai` 从 1.6.0 降到 1.1.9、`huggingface_hub` 降到 2.4.0，可能影响其他课程环境。因此按 RAGAS 口径实现了：

- Context Precision
- Correctness（LLM 裁判）
- Faithfulness（LLM 裁判）

### 7. Gradio 工作线程原生崩溃

现象：发送 Agent 请求后进程直接退出，退出码 `-1073740791`（0xC0000409），无 Python 堆栈。

修复：

- `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`KMP_DUPLICATE_LIB_OK=TRUE`
- `faiss.omp_set_num_threads(1)`
- 启动时在主线程预热 embedding / reranker，避免工作线程首次初始化 CUDA
- `demo.queue(default_concurrency_limit=1)`

### 8. 多轮 Agent 隔离

工具 Agent 只接收当前轮问题，历史由 `contextualize` 节点改写为独立问题：

```
“他的直属主管是谁？” → “张伟（工号 E1024）的直属主管是谁？”
```

这样既保留指代能力，又避免工具 Agent 被历史回答带偏、重复调用旧工具。

## 日志

运行日志写入 `logs/app.log`，按 2MB 轮转，保留 3 份。包含：

- 文档新增/删除/重建
- RAG 检索、精排、生成耗时
- Agent 路由、工具调用次数、总耗时

## 后续可扩展

- 向量持久化到 SQLite BLOB，删除文档时避免重新向量化
- 文档级权限过滤
- Agent 轨迹导出与回放
- 接入 RAGAS 独立环境做交叉验证
- 支持图片 PDF 的 OCR 入库
