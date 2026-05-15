import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

print("正在加载Embedding模型...")

model = SentenceTransformer(EMBEDDING_MODEL)

print("Embedding模型加载完成")