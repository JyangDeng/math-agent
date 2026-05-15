# config.py
import os

# 获取配置文件所在目录（项目根目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# 路径配置
# =========================

KNOWLEDGE_DIR = os.path.join(BASE_DIR, "rag/knowledge")

INDEX_DIR = os.path.join(BASE_DIR, "rag/index")

FAISS_INDEX_PATH = os.path.join(BASE_DIR, "rag/index/faiss.index")

DOCUMENTS_PATH = os.path.join(BASE_DIR, "rag/index/documents.pkl")

# =========================
# Embedding模型
# =========================
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# =========================
# LLM模型
# =========================
LLM_MODEL = "qwen3.6-max-preview"

# =========================
# Agent配置
# =========================
MAX_STEPS = 25
DEBUG = False