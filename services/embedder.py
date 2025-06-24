from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from .config import EMBEDDING_MODEL, BATCH_SIZE, NORMALIZE

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed(self, chunks):
        all_embeddings = []
        for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding"):
            batch = chunks[i:i+BATCH_SIZE]
            emb = self.model.encode(
                batch,
                batch_size=BATCH_SIZE,
                convert_to_numpy=True,
                normalize_embeddings=NORMALIZE,
                show_progress_bar=False
            )
            all_embeddings.append(emb)
        return np.vstack(all_embeddings).astype("float32")

    def embed_query(self, query: str):
        return self.model.encode([query], normalize_embeddings=NORMALIZE).astype("float32")
