"""Gradio 界面：RAG 对话 + 检索调试 + LangGraph Agent。"""
from __future__ import annotations

import html
import sys
import uuid
from pathlib import Path

# 支持 python app/ui.py 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr

from app.agent.runner import AgentRunner
from app.agent.state import AgentAnswer
from app.config import SERVER_NAME, SERVER_PORT
from app.ingest.pipeline import IngestionService
from app.logging_utils import setup_logging
from app.rag.pipeline import RagAnswer, RagPipeline
from app.retrieval.hybrid import RetrievalResult

setup_logging()

PIPELINE = RagPipeline()
if PIPELINE.retriever:
    PIPELINE.retriever.warmup()
AGENT = AgentRunner(retriever=PIPELINE.retriever) if PIPELINE.retriever else None

EMPTY_REFS = "<div style='color:#888;font-size:13px'>暂无引用来源</div>"


def _render_references(result: RagAnswer | None) -> str:
    if result is None or not result.hits:
        return EMPTY_REFS

    cards = []
    for index, hit in enumerate(result.hits, start=1):
        page = f" · 第 {hit.page} 页" if hit.page else ""
        snippet = hit.content.split("\n", 1)[-1][:180]
        rerank = f"{hit.rerank_score:.3f}" if hit.rerank_score is not None else "-"
        cards.append(
            f"""
            <div style="border:1px solid #e5e7eb;border-radius:10px;padding:10px 12px;margin-bottom:10px;background:#fff">
              <div style="font-size:13px;font-weight:600;color:#111827">
                [{index}] 《{html.escape(hit.doc_name)}》
                <span style="font-weight:400;color:#6b7280">/ {html.escape(hit.section)}{page}</span>
              </div>
              <div style="font-size:12px;color:#2563eb;margin:4px 0">精排 {rerank} · RRF {hit.rrf_score:.4f}</div>
              <div style="font-size:12.5px;color:#374151;line-height:1.6">{html.escape(snippet)}…</div>
            </div>
            """
        )

    queries = ""
    if len(result.queries) > 1:
        queries = "<div style='font-size:12px;color:#6b7280;margin-bottom:8px'>改写查询：" + " ｜ ".join(
            html.escape(item) for item in result.queries
        ) + "</div>"

    footer = (
        f"<div style='font-size:12px;color:#9ca3af;margin-top:6px'>"
        f"检索 {len(result.hits)} 条 · 模型 {html.escape(result.model)} · "
        f"检索 {result.retrieval_elapsed:.2f}s · 精排 {result.rerank_elapsed:.2f}s · "
        f"总耗时 {result.elapsed:.1f}s</div>"
    )
    return queries + "".join(cards) + footer


def chat_fn(message: str, history: list[dict], use_rewrite: bool):
    message = (message or "").strip()
    history = history or []
    if not message:
        return "", history, EMPTY_REFS

    try:
        result = PIPELINE.answer(message, history, use_rewrite=use_rewrite)
        answer = result.answer
    except Exception as exc:  # noqa: BLE001 - 界面需要展示错误
        result = None
        if "429" in str(exc) or "限流" in str(exc):
            answer = "模型服务当前触发限流（429），已自动重试仍未恢复，请稍等 1 分钟再提问。"
        else:
            answer = f"调用失败：{exc}"

    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return "", new_history, _render_references(result)


def clear_fn():
    return [], EMPTY_REFS


def reload_fn():
    PIPELINE.reload()
    return PIPELINE.status_text()


def _doc_choices() -> list[str]:
    return [row["doc_name"] for row in PIPELINE.store.list_documents()]


def _doc_table() -> str:
    rows = PIPELINE.store.list_documents()
    if not rows:
        return "<div style='color:#888;font-size:13px'>知识库为空</div>"

    body = "".join(
        f"<tr><td>《{html.escape(row['doc_name'])}》</td>"
        f"<td>{html.escape(row['file_type'])}</td>"
        f"<td>{row['chunk_count']}</td><td>{row['char_count']}</td></tr>"
        for row in rows
    )
    return f"""
    <div style="max-height:360px;overflow:auto">
      <table class="debug-table">
        <thead><tr><th>文档</th><th>类型</th><th>切片</th><th>字符数</th></tr></thead>
        <tbody>{body}</tbody>
      </table>
    </div>
    """


def _refresh_services():
    global AGENT
    PIPELINE.reload()
    if PIPELINE.retriever:
        PIPELINE.retriever.warmup()
    AGENT = AgentRunner(retriever=PIPELINE.retriever) if PIPELINE.retriever else None
    return (
        gr.update(choices=_doc_choices(), value=None),
        _doc_table(),
        PIPELINE.status_text(),
    )


def upload_docs(files):
    if not files:
        return "请选择要上传的文件", gr.update(), _doc_table(), PIPELINE.status_text()

    service = IngestionService()
    messages = []
    for file in files:
        path = Path(getattr(file, "path", file))
        try:
            result = service.add_file(path)
            messages.append(f"已入库《{result.doc_name}》：{result.chunks} 个切片")
        except Exception as exc:  # noqa: BLE001 - 单个文件失败不影响其他文件
            messages.append(f"《{path.name}》失败：{exc}")

    choices_update, table, status = _refresh_services()
    return "\n\n".join(messages), choices_update, table, status


def delete_doc(doc_name: str):
    if not doc_name:
        return "请选择要删除的文档", gr.update(), _doc_table(), PIPELINE.status_text()
    try:
        info = IngestionService().remove_document(doc_name)
        message = f"已删除《{doc_name}》，移除 {info['deleted_chunks']} 个切片"
    except Exception as exc:  # noqa: BLE001
        message = f"删除失败：{exc}"

    choices_update, table, status = _refresh_services()
    return message, choices_update, table, status


def rebuild_docs():
    try:
        info = IngestionService().rebuild()
        message = f"重建完成：{info['docs']} 篇文档，{info['chunks']} 个切片"
    except Exception as exc:  # noqa: BLE001
        message = f"重建失败：{exc}"

    choices_update, table, status = _refresh_services()
    return message, choices_update, table, status


def refresh_docs():
    choices_update, table, status = _refresh_services()
    return "已刷新", choices_update, table, status


def _debug_table(result: RetrievalResult) -> str:
    if not result.candidates:
        return "<div style='color:#888'>没有检索结果</div>"

    final_positions = {hit.chunk.faiss_pos for hit in result.hits}
    rows = []
    for rank, hit in enumerate(result.candidates[:20], start=1):
        is_final = hit.chunk.faiss_pos in final_positions
        dense = f"{hit.dense_rank} / {hit.dense_score:.3f}" if hit.dense_rank else "-"
        bm25 = f"{hit.bm25_rank} / {hit.bm25_score:.2f}" if hit.bm25_rank else "-"
        rerank = f"{hit.rerank_score:.4f}" if hit.rerank_score is not None else "-"
        snippet = html.escape(hit.content.split("\n", 1)[-1][:90])
        background = "#eef6ff" if is_final else "#fff"
        badge = "入 top5" if is_final else ""
        rows.append(
            f"""
            <tr style="background:{background}">
              <td>{rank}</td>
              <td>《{html.escape(hit.doc_name)}》<br><span class="muted">{html.escape(hit.section)}</span></td>
              <td>{hit.page or '-'}</td>
              <td>{dense}</td>
              <td>{bm25}</td>
              <td>{hit.rrf_score:.4f}</td>
              <td>{rerank}</td>
              <td class="badge">{badge}</td>
              <td class="snippet">{snippet}…</td>
            </tr>
            """
        )

    return f"""
    <div style="max-height:520px;overflow:auto">
    <table class="debug-table">
      <thead>
        <tr>
          <th>#</th><th>文档 / 章节</th><th>页</th>
          <th>向量 排名/分数</th><th>BM25 排名/分数</th>
          <th>RRF</th><th>精排</th><th></th><th>片段</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    </div>
    <div style="font-size:12px;color:#6b7280;margin-top:8px">
      共 {len(result.candidates)} 个融合候选 · 检索 {result.elapsed:.2f}s · 精排 {result.rerank_elapsed:.2f}s
    </div>
    """


def debug_fn(query: str, use_rewrite: bool, use_rerank: bool):
    query = (query or "").strip()
    if not query:
        return "请输入查询语句", "<div style='color:#888'>等待检索</div>"

    try:
        result = PIPELINE.retrieve(query, use_rewrite=use_rewrite, use_rerank=use_rerank)
        queries = "**实际检索查询：**\n" + "\n".join(f"- {item}" for item in result.queries)
        return queries, _debug_table(result)
    except Exception as exc:  # noqa: BLE001
        return f"检索失败：{exc}", ""


ROUTE_LABELS = {
    "knowledge": ("知识库检索", "#2563eb", "#eff6ff"),
    "tools": ("工具调用", "#b45309", "#fffbeb"),
    "direct": ("直接回答", "#4b5563", "#f9fafb"),
}


def _render_agent_trace(result: AgentAnswer | None) -> str:
    if result is None:
        return "<div style='color:#888;font-size:13px'>暂无执行轨迹</div>"

    label, color, background = ROUTE_LABELS.get(result.route, (result.route, "#374151", "#f9fafb"))
    rewritten = ""
    if result.question and result.question != result.original_question:
        rewritten = (
            "<div class='agent-note'>上下文改写："
            f"{html.escape(result.original_question)} → <b>{html.escape(result.question)}</b></div>"
        )

    steps = "".join(
        f"<div class='agent-step'><span>{index}</span>{html.escape(step)}</div>"
        for index, step in enumerate(result.steps, start=1)
    )

    tool_cards = []
    for item in result.tool_calls:
        if item.get("type") == "call":
            args = ", ".join(f"{key}={value}" for key, value in (item.get("args") or {}).items())
            tool_cards.append(
                f"<div class='tool-card call'>调用 <b>{html.escape(str(item.get('name')))}</b>"
                f"<div class='tool-args'>{html.escape(args)}</div></div>"
            )
        else:
            tool_cards.append(
                f"<div class='tool-card result'>返回 <b>{html.escape(str(item.get('name')))}</b>"
                f"<div class='tool-result'>{html.escape(str(item.get('content', ''))[:260])}…</div></div>"
            )
    tools_html = "".join(tool_cards) or "<div style='color:#9ca3af;font-size:12px'>本轮未调用工具</div>"

    citation_cards = []
    for index, citation in enumerate(result.citations, start=1):
        page = f" · 第 {citation['page']} 页" if citation.get("page") else ""
        citation_cards.append(
            f"<div class='cite-card'>[{index}] 《{html.escape(str(citation['doc_name']))}》"
            f"<span> / {html.escape(str(citation['section']))}{page}</span>"
            f"<em>精排 {float(citation.get('score', 0)):.3f}</em></div>"
        )
    citations_html = "".join(citation_cards) or "<div style='color:#9ca3af;font-size:12px'>无引用</div>"

    return f"""
    <div class="agent-header" style="background:{background};border-color:{color}">
      <span class="agent-badge" style="background:{color}">{label}</span>
      <span class="agent-reason">{html.escape(result.route_reason)}</span>
      <span class="agent-time">{result.elapsed:.1f}s</span>
    </div>
    {rewritten}
    <div class="agent-section"><div class="agent-title">执行轨迹</div>{steps}</div>
    <div class="agent-section"><div class="agent-title">工具调用</div>{tools_html}</div>
    <div class="agent-section"><div class="agent-title">引用来源</div>{citations_html}</div>
    """


def agent_chat_fn(message: str, history: list[dict], thread_id: str):
    message = (message or "").strip()
    history = history or []
    if not message:
        return "", history, "<div style='color:#888;font-size:13px'>暂无执行轨迹</div>"

    if AGENT is None:
        return "", history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "索引未加载，请先运行 scripts/build_index.py。"},
        ], "<div style='color:#888'>Agent 不可用</div>"

    try:
        result = AGENT.run(message, thread_id=thread_id or "default")
        answer = result.answer
    except Exception as exc:  # noqa: BLE001
        result = None
        answer = f"Agent 执行失败：{exc}"

    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return "", new_history, _render_agent_trace(result)


def new_agent_session():
    return str(uuid.uuid4()), [], "<div style='color:#888;font-size:13px'>新会话已开始</div>"


DEBUG_CSS = """
.debug-table { width:100%; border-collapse:collapse; font-size:12.5px; }
.debug-table th { position:sticky; top:0; background:#f3f4f6; text-align:left; padding:8px; border-bottom:1px solid #e5e7eb; }
.debug-table td { padding:8px; border-bottom:1px solid #f0f0f0; vertical-align:top; }
.debug-table .muted { color:#6b7280; font-size:12px; }
.debug-table .badge { color:#2563eb; font-size:12px; }
.debug-table .snippet { color:#374151; max-width:420px; }

.agent-header { display:flex; align-items:center; gap:8px; padding:10px 12px; border:1px solid; border-radius:10px; margin-bottom:10px; }
.agent-badge { color:#fff; font-size:12px; padding:3px 8px; border-radius:999px; }
.agent-reason { flex:1; font-size:12px; color:#374151; }
.agent-time { font-size:12px; color:#6b7280; }
.agent-note { font-size:12px; color:#6b7280; background:#f9fafb; border-radius:8px; padding:8px 10px; margin-bottom:10px; }
.agent-section { margin-bottom:14px; }
.agent-title { font-size:12px; font-weight:600; color:#111827; margin-bottom:6px; }
.agent-step { font-size:12px; color:#374151; padding:5px 0; border-bottom:1px dashed #f0f0f0; }
.agent-step span { display:inline-block; width:18px; height:18px; line-height:18px; text-align:center; border-radius:50%; background:#eef2ff; color:#4338ca; font-size:11px; margin-right:8px; }
.tool-card { border:1px solid #e5e7eb; border-radius:8px; padding:8px 10px; margin-bottom:6px; font-size:12px; }
.tool-card.call { background:#fffbeb; border-color:#fde68a; }
.tool-card.result { background:#f9fafb; }
.tool-args { color:#6b7280; margin-top:4px; }
.tool-result { color:#374151; margin-top:4px; white-space:pre-wrap; }
.cite-card { font-size:12px; padding:6px 0; border-bottom:1px dashed #f0f0f0; color:#111827; }
.cite-card span { color:#6b7280; }
.cite-card em { float:right; color:#2563eb; font-style:normal; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="星澜科技 · 企业知识库问答") as demo:
        gr.Markdown(
            "## 星澜科技 · 企业知识库助手\n"
            "RAG：bge-large-zh + BM25 → RRF → bge-reranker-base；Agent：LangGraph 路由 + 工具调用 + 检索评分循环。"
        )

        with gr.Tabs():
            with gr.Tab("对话"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(label="对话", height=540)
                        with gr.Row():
                            textbox = gr.Textbox(
                                placeholder="例如：出差到上海，住宿和补贴标准是多少？",
                                show_label=False,
                                scale=5,
                            )
                            send_btn = gr.Button("发送", variant="primary", scale=1)
                        with gr.Row():
                            chat_rewrite = gr.Checkbox(label="启用 Query 改写", value=False, scale=1)
                            clear_btn = gr.Button("清空对话", scale=1)
                            reload_btn = gr.Button("重载索引", scale=1)

                    with gr.Column(scale=2):
                        status = gr.Markdown(PIPELINE.status_text())
                        gr.Markdown("### 引用来源")
                        references = gr.HTML(EMPTY_REFS)

                send_btn.click(
                    chat_fn,
                    [textbox, chatbot, chat_rewrite],
                    [textbox, chatbot, references],
                )
                textbox.submit(
                    chat_fn,
                    [textbox, chatbot, chat_rewrite],
                    [textbox, chatbot, references],
                )
                clear_btn.click(clear_fn, None, [chatbot, references])
                reload_btn.click(reload_fn, None, [status])

            with gr.Tab("检索调试"):
                gr.Markdown("查看向量、BM25、RRF 和精排四层结果，蓝色行表示最终进入 top5 的片段。")
                debug_query = gr.Textbox(label="查询", placeholder="例如：P0 工单多久响应？")
                with gr.Row():
                    debug_rewrite = gr.Checkbox(label="启用 Query 改写", value=False)
                    debug_rerank = gr.Checkbox(label="启用精排", value=True)
                    debug_btn = gr.Button("执行检索", variant="primary")
                debug_queries = gr.Markdown("**实际检索查询：** -")
                debug_table = gr.HTML("<div style='color:#888'>等待检索</div>")
                gr.Examples(
                    examples=[
                        "P0 工单多久响应？",
                        "税号 91330100MA2K7X8Y9Z 是什么？",
                        "出差住宿和补贴怎么算，发票抬头是什么？",
                        "年假有多少天？",
                    ],
                    inputs=debug_query,
                )
                debug_btn.click(
                    debug_fn,
                    [debug_query, debug_rewrite, debug_rerank],
                    [debug_queries, debug_table],
                )
                debug_query.submit(
                    debug_fn,
                    [debug_query, debug_rewrite, debug_rerank],
                    [debug_queries, debug_table],
                )

            with gr.Tab("Agent"):
                gr.Markdown(
                    "LangGraph 流程：上下文改写 → 路由 → 检索 → 资料评分 → 改写重试 / 生成；"
                    "工具类问题交给 `create_agent` 执行。右侧可看路由、轨迹和工具调用。"
                )
                agent_thread = gr.State("default")
                with gr.Row():
                    with gr.Column(scale=3):
                        agent_chatbot = gr.Chatbot(label="Agent 对话", height=540)
                        with gr.Row():
                            agent_textbox = gr.Textbox(
                                placeholder="例如：帮我算去上海出差 3 天的预算",
                                show_label=False,
                                scale=5,
                            )
                            agent_send = gr.Button("发送", variant="primary", scale=1)
                        with gr.Row():
                            agent_new = gr.Button("新会话")
                            agent_clear = gr.Button("清空显示")

                    with gr.Column(scale=2):
                        gr.Markdown("### 执行轨迹")
                        agent_trace = gr.HTML("<div style='color:#888;font-size:13px'>暂无执行轨迹</div>")

                gr.Examples(
                    examples=[
                        "P0 工单多久响应？",
                        "帮我算下去上海出差 3 天的预算",
                        "张伟是哪个部门的？",
                        "知识库里有多少文档和切片？",
                        "公司有没有健身房？",
                    ],
                    inputs=agent_textbox,
                )

                agent_send.click(
                    agent_chat_fn,
                    [agent_textbox, agent_chatbot, agent_thread],
                    [agent_textbox, agent_chatbot, agent_trace],
                )
                agent_textbox.submit(
                    agent_chat_fn,
                    [agent_textbox, agent_chatbot, agent_thread],
                    [agent_textbox, agent_chatbot, agent_trace],
                )
                agent_new.click(
                    new_agent_session,
                    None,
                    [agent_thread, agent_chatbot, agent_trace],
                )
                agent_clear.click(clear_fn, None, [agent_chatbot, agent_trace])
                demo.load(lambda: str(uuid.uuid4()), None, [agent_thread])

            with gr.Tab("文档管理"):
                gr.Markdown(
                    "上传 Markdown / PDF / Word 文档后会自动解析、切分、向量化并增量入库；"
                    "删除文档会重建剩余索引。"
                )
                upload_files = gr.File(
                    label="选择文档",
                    file_count="multiple",
                    file_types=[".md", ".markdown", ".pdf", ".docx"],
                )
                with gr.Row():
                    upload_btn = gr.Button("上传并入库", variant="primary")
                    rebuild_btn = gr.Button("全量重建")
                    refresh_btn = gr.Button("刷新")

                ingest_status = gr.Markdown("等待操作")
                doc_table = gr.HTML(_doc_table())
                delete_name = gr.Dropdown(
                    choices=_doc_choices(),
                    label="选择要删除的文档",
                )
                delete_btn = gr.Button("删除文档")

                upload_btn.click(
                    upload_docs,
                    [upload_files],
                    [ingest_status, delete_name, doc_table, status],
                )
                delete_btn.click(
                    delete_doc,
                    [delete_name],
                    [ingest_status, delete_name, doc_table, status],
                )
                rebuild_btn.click(
                    rebuild_docs,
                    None,
                    [ingest_status, delete_name, doc_table, status],
                )
                refresh_btn.click(
                    refresh_docs,
                    None,
                    [ingest_status, delete_name, doc_table, status],
                )

    # 单并发，避免多个请求同时进入向量/精排模型引发原生库线程冲突
    demo.queue(default_concurrency_limit=1)
    return demo


if __name__ == "__main__":
    build_ui().launch(
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
        inbrowser=True,
        css=DEBUG_CSS,
    )
