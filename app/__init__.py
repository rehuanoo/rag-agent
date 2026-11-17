"""企业知识库 RAG Agent。

在导入 torch / faiss 之前限制底层线程数，避免 Windows 下 Gradio 工作线程
触发 OpenMP 原生崩溃。
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
