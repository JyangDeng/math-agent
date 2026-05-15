import os
import faiss
import pickle
import numpy as np
import sys

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer

# 获取当前文件所在目录（rag/）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录（math-agent/）
parent_dir = os.path.dirname(current_dir)
# 添加到Python模块搜索路径
sys.path.insert(0, parent_dir)

# 现在可以导入config
from config import KNOWLEDGE_DIR, INDEX_DIR, FAISS_INDEX_PATH, DOCUMENTS_PATH, EMBEDDING_MODEL
# =====================================
# 加载Embedding模型
# =====================================

print("正在加载Embedding模型...")

model = SentenceTransformer(
    EMBEDDING_MODEL
)

# =====================================
# 读取知识库
# =====================================

knowledge_dir = KNOWLEDGE_DIR

documents = []

for filename in os.listdir(knowledge_dir):

    path = os.path.join(
        knowledge_dir,
        filename
    )

    if filename.endswith(".md"):

        with open(path, "r", encoding="utf-8") as f:

            text = f.read()

            documents.append(text)

print(f"读取文档数量: {len(documents)}")

# =====================================
# 向量化
# =====================================

print("正在生成向量...")

embeddings = model.encode(documents)

embeddings = np.array(
    embeddings
).astype("float32")

# =====================================
# 创建FAISS索引
# =====================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

# =====================================
# 创建 index 文件夹
# =====================================

os.makedirs(
    INDEX_DIR,
    exist_ok=True
)

# =====================================
# 保存索引
# =====================================

faiss.write_index(
    index,
    FAISS_INDEX_PATH
)

# 保存原始文档

with open(
    DOCUMENTS_PATH,
    "wb"
) as f:

    pickle.dump(documents, f)

print("知识库索引构建完成！")