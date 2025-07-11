{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d75564e3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🚀 Starting optimized RAG system...\n",
      "🔄 Loading embedding model...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\kau75421\\AppData\\Local\\Temp\\ipykernel_40860\\3310735168.py:41: LangChainDeprecationWarning: The class `Ollama` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the :class:`~langchain-ollama package and should be used instead. To use it run `pip install -U :class:`~langchain-ollama` and import as `from :class:`~langchain_ollama import OllamaLLM``.\n",
      "  llm = Ollama(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🔄 Loading FAISS index...\n",
      "🔄 Loading LLM...\n",
      "✅ All models loaded! Ready for queries.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "import faiss\n",
    "import pickle\n",
    "import numpy as np\n",
    "from sentence_transformers import SentenceTransformer\n",
    "from langchain_community.llms import Ollama\n",
    "from langchain_core.messages import HumanMessage, SystemMessage\n",
    "import time\n",
    "\n",
    "# --- OPTIMIZED CONFIG ---\n",
    "FAISS_INDEX_PATH = r\"C:\\Users\\kau75421\\LLMprojects\\Marketing_campaginer\\Recommender_Systems\\Notebooks\\faiss.index\"\n",
    "CHUNKS_MAPPING_PATH = r\"C:\\Users\\kau75421\\LLMprojects\\Marketing_campaginer\\Recommender_Systems\\Notebooks\\faiss_data.pkl\"\n",
    "EMBEDDING_MODEL = 'all-MiniLM-L6-v2'\n",
    "NORMALIZE = True\n",
    "TOP_K = 3                             # Reduced from 5\n",
    "SCORE_THRESHOLD = 0.2                 # Lowered for more matches\n",
    "OLLAMA_MODEL = \"mistral:7b-instruct-q4_0\"\n",
    "\n",
    "# --- GLOBAL VARIABLES FOR CACHING ---\n",
    "embedder = None\n",
    "index = None\n",
    "chunk_mapping = None\n",
    "llm = None\n",
    "\n",
    "# --- OPTIMIZED UTILS ---\n",
    "def load_models_once():\n",
    "    \"\"\"Load all models once at startup\"\"\"\n",
    "    global embedder, index, chunk_mapping, llm\n",
    "    \n",
    "    if embedder is None:\n",
    "        print(\"🔄 Loading embedding model...\")\n",
    "        embedder = SentenceTransformer(EMBEDDING_MODEL)\n",
    "        \n",
    "    if index is None or chunk_mapping is None:\n",
    "        print(\"🔄 Loading FAISS index...\")\n",
    "        index = faiss.read_index(FAISS_INDEX_PATH)\n",
    "        with open(CHUNKS_MAPPING_PATH, \"rb\") as f:\n",
    "            chunk_mapping = pickle.load(f)\n",
    "            \n",
    "    if llm is None:\n",
    "        print(\"🔄 Loading LLM...\")\n",
    "        llm = Ollama(\n",
    "            model=OLLAMA_MODEL, \n",
    "            temperature=0.0,\n",
    "            # Add these for faster inference\n",
    "            num_predict=200,  # Limit response length\n",
    "            top_k=10,         # Reduce sampling space\n",
    "            top_p=0.9\n",
    "        )\n",
    "\n",
    "def embed_query_fast(query):\n",
    "    \"\"\"Faster query embedding\"\"\"\n",
    "    return embedder.encode([query], normalize_embeddings=NORMALIZE, show_progress_bar=False).astype(\"float32\")\n",
    "\n",
    "def retrieve_and_filter(query_embedding, k=TOP_K):\n",
    "    \"\"\"Combined retrieval and filtering\"\"\"\n",
    "    distances, indices = index.search(query_embedding, k)\n",
    "    \n",
    "    # Quick filtering and deduplication in one pass\n",
    "    seen = set()\n",
    "    filtered_chunks = []\n",
    "    \n",
    "    for i, (idx, score) in enumerate(zip(indices[0], distances[0])):\n",
    "        if score >= SCORE_THRESHOLD:\n",
    "            chunk = chunk_mapping[idx]\n",
    "            if chunk not in seen:\n",
    "                seen.add(chunk)\n",
    "                filtered_chunks.append(chunk)\n",
    "                \n",
    "    return filtered_chunks\n",
    "\n",
    "def build_concise_prompt(chunks, user_query):\n",
    "    \"\"\"Shorter prompt for faster processing\"\"\"\n",
    "    # Take only top 2 chunks and truncate them\n",
    "    context_chunks = []\n",
    "    for chunk in chunks[:2]:\n",
    "        # Truncate long chunks\n",
    "        truncated = chunk[:300] + \"...\" if len(chunk) > 300 else chunk\n",
    "        context_chunks.append(truncated)\n",
    "    \n",
    "    context = \"\\n---\\n\".join(context_chunks)\n",
    "    \n",
    "    return f\"\"\"Based on this product info, answer briefly:\n",
    "\n",
    "{context}\n",
    "\n",
    "Q: {user_query}\n",
    "A:\"\"\"\n",
    "\n",
    "def get_llm_response_fast(prompt):\n",
    "    \"\"\"Faster LLM response handling\"\"\"\n",
    "    try:\n",
    "        # Use simpler message format\n",
    "        response = llm.invoke(prompt)\n",
    "        \n",
    "        if isinstance(response, str):\n",
    "            return response.strip()\n",
    "        elif hasattr(response, 'content'):\n",
    "            return response.content.strip()\n",
    "        else:\n",
    "            return str(response).strip()\n",
    "            \n",
    "    except Exception as e:\n",
    "        return f\"⚠️ Error getting response: {str(e)}\"\n",
    "\n",
    "# --- OPTIMIZED MAIN ---\n",
    "def main():\n",
    "    print(\"🚀 Starting optimized RAG system...\")\n",
    "    \n",
    "    # Load everything once at startup\n",
    "    load_models_once()\n",
    "    print(\"✅ All models loaded! Ready for queries.\\n\")\n",
    "\n",
    "    while True:\n",
    "        user_query = input(\"🔍 Ask about products (or 'exit'): \").strip()\n",
    "        if user_query.lower() in ['exit', 'quit', 'q']:\n",
    "            break\n",
    "\n",
    "        if len(user_query) < 3:\n",
    "            print(\"⚠️ Please enter a longer query.\")\n",
    "            continue\n",
    "\n",
    "        start_time = time.time()\n",
    "\n",
    "        # Step 1: Fast embedding\n",
    "        query_embedding = embed_query_fast(user_query)\n",
    "        embed_time = time.time()\n",
    "\n",
    "        # Step 2: Fast retrieval and filtering\n",
    "        chunks = retrieve_and_filter(query_embedding)\n",
    "        retrieval_time = time.time()\n",
    "\n",
    "        if not chunks:\n",
    "            print(\"⚠️ No relevant products found. Try a different query.\")\n",
    "            continue\n",
    "\n",
    "        # Step 3: Build concise prompt\n",
    "        prompt = build_concise_prompt(chunks, user_query)\n",
    "        \n",
    "        # Optional: Show retrieved info (comment out for even faster performance)\n",
    "        print(f\"\\n📄 Found {len(chunks)} relevant chunks\")\n",
    "        \n",
    "        # Step 4: Fast LLM response\n",
    "        print(\"\\n💬 Answer:\")\n",
    "        response = get_llm_response_fast(prompt)\n",
    "        print(response)\n",
    "        \n",
    "        llm_time = time.time()\n",
    "\n",
    "        # Timing breakdown\n",
    "        total_time = llm_time - start_time\n",
    "        print(f\"\\n⏱️ Timing: Embed: {embed_time-start_time:.2f}s | \"\n",
    "              f\"Retrieve: {retrieval_time-embed_time:.2f}s | \"\n",
    "              f\"LLM: {llm_time-retrieval_time:.2f}s | \"\n",
    "              f\"Total: {total_time:.2f}s\")\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "87f1c679",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
