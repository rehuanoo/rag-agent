"""项目配置：路径、模型、切分与检索参数。"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"
LOG_DIR = PROJECT_ROOT / "logs"
TRACE_DIR = DATA_DIR / "traces"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
META_DB_PATH = INDEX_DIR / "metadata.db"

# 本地向量模型（bge-large-zh-v1.5，1024 维，最大 512 token）
EMBEDDING_MODEL_PATH = Path(
    r"S:\PythonAILearningCourse\6.Transformers\models\bge-large-zh-v1.5"
)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
EMBEDDING_MAX_SEQ_LEN = 512
EMBEDDING_DIM = 1024

# DeepSeek，OpenAI 兼容接口
LLM_BASE_URL = "https://api.deepseek.com/v1"
LLM_PRIMARY_MODEL = "deepseek-flash"
LLM_FALLBACK_MODELS = ("deepseek-v4-pro",)
LLM_TEMPERATURE = 0.2
LLM_TIMEOUT = 60
LLM_MAX_TOKENS = 800
# 每轮把所有模型试一次；全部失败时按下面的间隔等待后重试
LLM_RETRY_DELAYS = (5, 15)

# 远端不可用时的本地兜底（当前本地只有 base 模型，默认关闭）
LOCAL_LLM_ENABLED = False
LOCAL_LLM_PATH = Path(r"S:\PythonAILearningCourse\6.Transformers\models\Qwen1.5-1.8B")
LOCAL_LLM_NAME = "Qwen1.5-1.8B（本地）"
LOCAL_LLM_MAX_NEW_TOKENS = 160
LOCAL_LLM_MAX_INPUT_TOKENS = 6144

# 切分与检索
CHUNK_SIZE = 400
CHUNK_OVERLAP = 60
TOP_K = 5
MIN_SCORE = 0.25
MAX_HISTORY_TURNS = 3

# 混合检索
BM25_TOP_K = 20
RRF_K = 60
RERANK_ENABLED = True
RERANK_CANDIDATES = 20
RERANK_BATCH_SIZE = 16
RERANKER_MODEL_PATH = Path(
    r"S:\PythonAILearningCourse\6.Transformers\models\bge-reranker-base"
)

# Query 改写
QUERY_REWRITE_MAX_QUERIES = 3

# LangGraph Agent
AGENT_MAX_REWRITES = 2

# Gradio
SERVER_NAME = "127.0.0.1"
SERVER_PORT = 7860


def resolve_api_key() -> str:
    """读取 DeepSeek API Key。

    优先使用进程环境变量；若未继承（例如 IDE 在变量设置前启动），
    再回退到 Windows 系统级环境变量 dsAPIkey。
    """
    names = ("dsAPIkey", "DEEPSEEK_API_KEY")
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()

    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
            ) as key:
                for name in names:
                    try:
                        value, _ = winreg.QueryValueEx(key, name)
                    except FileNotFoundError:
                        continue
                    if value:
                        return str(value).strip()
        except OSError:
            pass

    raise RuntimeError("未找到 DeepSeek API Key，请设置系统环境变量 dsAPIkey")
