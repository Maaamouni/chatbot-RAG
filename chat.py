import os
import sys
import requests
import chromadb

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:1.5b"
TOP_K = 6
MAX_CONTEXT_CHARS = 4000


class OllamaEmbeddingFunction:
    def __init__(self, model_name, url):
        self.model_name = model_name
        self.url = url

    def name(self):
        return f"ollama_{self.model_name}"

    def _embed(self, texts, prefix=""):
        if isinstance(texts, str):
            texts = [texts]
        if prefix:
            texts = [f"{prefix}{t}" for t in texts]
        response = requests.post(
            f"{self.url}/api/embed",
            json={"model": self.model_name, "input": texts},
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    def __call__(self, input):
        return self._embed(input)

    def embed_query(self, input):
        return self._embed(input, prefix="search_query: ")

    def embed_document(self, input):
        return self._embed(input, prefix="search_document: ")


def build_context(results):
    context_parts = []
    total_chars = 0
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        snippet = f"[{meta['path']}]\n{doc}"
        if total_chars + len(snippet) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining > 200:
                context_parts.append(snippet[:remaining])
            break
        context_parts.append(snippet)
        total_chars += len(snippet)
    return "\n\n---\n\n".join(context_parts)


def ask_question(query, collection, history):
    results = collection.query(
        query_texts=[query],
        n_results=TOP_K,
    )

    if not results["documents"] or not results["documents"][0]:
        print("No relevant notes found.")
        return ""

    context = build_context(results)

    sources = list(set(m["path"] for m in results["metadatas"][0]))
    print(f"\n[Sources: {', '.join(sources)}]")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a personal knowledge assistant. Answer the user's question "
                "based on their private notes provided as context. If the notes don't "
                "contain enough information, say so and provide your best general "
                "knowledge answer. Keep responses concise and accurate."
            ),
        },
    ]

    messages.extend(history[-6:])
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": messages,
            "stream": True,
        },
        stream=True,
    )
    response.raise_for_status()

    full_response = ""
    import json
    for line in response.iter_lines():
        if line:
            try:
                chunk = json.loads(line.decode("utf-8"))
                content = chunk.get("message", {}).get("content", "")
                if content:
                    print(content, end="", flush=True)
                    full_response += content
            except json.JSONDecodeError:
                pass

    return full_response


def main():
    if not os.path.exists(CHROMA_DIR):
        print("No index found. Run ingest.py first.")
        return

    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    except requests.ConnectionError:
        print(f"Cannot reach Ollama at {OLLAMA_URL}")
        print("Start it with: ollama serve")
        return

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(
        name="cherrytree_notes",
        embedding_function=OllamaEmbeddingFunction(EMBED_MODEL, OLLAMA_URL),
    )

    count = collection.count()
    print(f"Loaded {count} chunks from your notes.")
    print(f"Chat ready! (model: {CHAT_MODEL})\n")

    history = []

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        answer = ask_question(query, collection, history)
        if answer:
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": answer})
        print()
        return

    print("Interactive mode. Type 'quit' to exit, 'clear' to reset history.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            break
        if query.lower() == "clear":
            history = []
            print("History cleared.")
            continue

        print("\nBot: ", end="", flush=True)
        answer = ask_question(query, collection, history)
        if answer:
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": answer})
        print("\n")


if __name__ == "__main__":
    main()
