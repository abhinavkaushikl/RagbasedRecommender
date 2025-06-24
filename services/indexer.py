import faiss
import pickle
from tqdm import tqdm
from .config import FAISS_INDEX_PATH, CHUNKS_MAPPING_PATH

class Indexer:
    def __init__(self):
        self.index = None

    def build_index(self, embeddings):
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        for i in tqdm(range(0, len(embeddings), 10000), desc="Adding to FAISS"):
            self.index.add(embeddings[i:i+10000])

    def save(self, chunks):
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        with open(CHUNKS_MAPPING_PATH, "wb") as f:
            pickle.dump(chunks, f)

    def load(self):
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        with open(CHUNKS_MAPPING_PATH, "rb") as f:
            return self.index, pickle.load(f)
