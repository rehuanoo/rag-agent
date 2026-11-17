"""SenseNova 大模型客户端：远端优先，本地 Qwen 兜底。"""
from __future__ import annotations

import time
from typing import Sequence

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from app.config import (
    LLM_BASE_URL,
    LLM_FALLBACK_MODELS,
    LLM_MAX_TOKENS,
    LLM_PRIMARY_MODEL,
    LLM_RETRY_DELAYS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    LOCAL_LLM_ENABLED,
    LOCAL_LLM_NAME,
    resolve_api_key,
)
from app.local_llm import LocalChatModel

_ROLE_MAP = {"system": "system", "human": "user", "ai": "assistant"}


def create_chat_model(
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or LLM_PRIMARY_MODEL,
        base_url=LLM_BASE_URL,
        api_key=resolve_api_key(),
        temperature=LLM_TEMPERATURE if temperature is None else temperature,
        max_tokens=max_tokens or LLM_MAX_TOKENS,
        timeout=LLM_TIMEOUT,
        max_retries=1,
        streaming=False,
    )


def _build_model(model: str) -> ChatOpenAI:
    return create_chat_model(model=model)


def _is_retryable(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        return True
    return status_code == 429 or status_code >= 500


def _to_dicts(messages: Sequence[BaseMessage]) -> list[dict]:
    dicts = [
        {
            "role": _ROLE_MAP.get(message.type, "user"),
            "content": str(message.content),
        }
        for message in messages
    ]
    # 小模型自由生成容易编造，改为严格的原文摘录模式
    if dicts and dicts[0]["role"] == "system":
        dicts[0]["content"] = (
            "你是资料摘录助手。只能原样摘录【参考资料】中的句子，"
            "不得改写、不得补充资料外的内容。只输出最能回答问题的原文，最多两段。"
        )
    return dicts


def _invoke_remote(models: Sequence[str], messages: Sequence[BaseMessage]) -> tuple[str, str] | None:
    for model in models:
        try:
            response = _build_model(model).invoke(list(messages))
            return str(response.content).strip(), model
        except Exception as exc:  # noqa: BLE001 - 需要记录并降级
            if not _is_retryable(exc):
                raise RuntimeError(f"模型鉴权或请求错误：{exc}") from exc
    return None


def chat_with_fallback(messages: Sequence[BaseMessage]) -> tuple[str, str]:
    """远端主模型 -> 备用模型 -> 本地 Qwen -> 退避重试远端。"""
    models = (LLM_PRIMARY_MODEL, *LLM_FALLBACK_MODELS)

    result = _invoke_remote(models, messages)
    if result:
        return result

    if LOCAL_LLM_ENABLED:
        try:
            text = LocalChatModel.get().chat(_to_dicts(messages))
            if text:
                return text, LOCAL_LLM_NAME
        except Exception:  # noqa: BLE001 - 本地模型失败不应阻断后续重试
            pass

    for delay in LLM_RETRY_DELAYS:
        time.sleep(delay)
        result = _invoke_remote(models, messages)
        if result:
            return result

    raise RuntimeError("所有模型调用失败（远端限流且本地模型不可用）")
