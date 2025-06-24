class Searcher:
    def __init__(self, index, chunk_mapping):
        self.index = index
        self.chunk_mapping = chunk_mapping

    def search(self, query_embedding, k=10):
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for i in range(k):
            idx = indices[0][i]
            score = distances[0][i]
            results.append((self.chunk_mapping[idx], score))
        return results
