from chunker import Chunker
from embedder import Embedder
from indexer import Indexer
from searcher import Searcher
from config import FAISS_INDEX_PATH
import os

def main():
    # Step 1: Chunk products
    chunker = Chunker()
    chunk_lists = chunker.to_chunks()
    flat_chunks = [chunk for sublist in chunk_lists for chunk in sublist]

    # Step 2: Embed
    embedder = Embedder()
    embeddings = embedder.embed(flat_chunks)

    # Step 3: Index & Save
    indexer = Indexer()
    indexer.build_index(embeddings)
    indexer.save(flat_chunks)

    # Step 4: Load for search
    index, chunk_mapping = indexer.load()
    searcher = Searcher(index, chunk_mapping)

    # Step 5: Search
    while True:
        query = input("🔍 Enter query or 'exit': ").strip()
        if query.lower() == "exit":
            break
        instruction = f"Retrieve product info: {query}"
        query_emb = embedder.embed_query(instruction)
        results = searcher.search(query_emb)

        for i, (chunk, score) in enumerate(results):
            print(f"\nResult {i+1} — Score: {score:.4f}")
            print(chunk)

if __name__ == "__main__":
    main()
