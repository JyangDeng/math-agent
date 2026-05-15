import faiss
import numpy as np

from config import (
    FAISS_INDEX_PATH,
    DOCUMENTS_PATH
)

index = faiss.read_index(FAISS_INDEX_PATH)

documents = np.load(
    DOCUMENTS_PATH,
    allow_pickle=True
)