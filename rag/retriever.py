import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from config import FAISS_INDEX_PATH,DOCUMENTS_PATH,EMBEDDING_MODEL

# =====================================
# 延迟加载模型
# =====================================

model = None

def get_model():

    global model

    if model is None:

        print("正在加载Embedding模型...")

        model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    return model

# =====================================
# 加载索引
# =====================================

index = faiss.read_index(FAISS_INDEX_PATH)

with open(DOCUMENTS_PATH, "rb") as f:
    documents = pickle.load(f)

# =====================================
# 检索函数
# =====================================

def retrieve_knowledge(query, top_k=3):

    model = get_model()

    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for idx in indices[0]:

        if idx < len(documents):

            results.append(documents[idx])

    return results