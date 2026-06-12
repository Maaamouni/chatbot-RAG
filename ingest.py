import os
import xml.etree.ElementTree as ET
import requests
import chromadb
from tqdm import tqdm

DATA_DIR = "/home/outhmane/Desktop/outhmanes_stuff"
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

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


def parse_node_xml(node_path):
    xml_file = os.path.join(node_path, "node.xml")
    if not os.path.exists(xml_file):
        return None, ""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    node_elem = root.find("node")
    if node_elem is None:
        return None, ""
    name = node_elem.get("name", "")
    texts = []
    for rich_text in node_elem.findall("rich_text"):
        if rich_text.text:
            texts.append(rich_text.text.strip())
    return name, "\n".join(texts)


def read_subnodes(node_path):
    subnodes_file = os.path.join(node_path, "subnodes.lst")
    if not os.path.exists(subnodes_file):
        return []
    with open(subnodes_file) as f:
        line = f.readline().strip()
    return [int(x.strip()) for x in line.split(",") if x.strip()]


def walk_tree():
    nodes = []

    def walk(current_path, parent_path_str):
        name, content = parse_node_xml(current_path)
        node_id = os.path.basename(current_path)
        path_str = f"{parent_path_str} / {name}" if parent_path_str else name
        if content.strip():
            nodes.append({
                "id": node_id,
                "name": name,
                "path": path_str,
                "content": content.strip(),
            })
        children = read_subnodes(current_path)
        for child_id in children:
            child_path = os.path.join(current_path, str(child_id))
            if os.path.exists(child_path):
                walk(child_path, path_str)

    root_children = read_subnodes(DATA_DIR)
    for child_id in root_children:
        child_path = os.path.join(DATA_DIR, str(child_id))
        if os.path.exists(child_path):
            walk(child_path, "")

    return nodes


def chunk_content(content, max_chars=1000, overlap=100):
    if len(content) <= max_chars:
        return [content]
    chunks = []
    start = 0
    while start < len(content):
        end = min(start + max_chars, len(content))
        if end < len(content):
            split_at = content.rfind("\n", start, end)
            if split_at > start + max_chars // 2:
                end = split_at
        chunks.append(content[start:end])
        if end >= len(content):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def main():
    print("Walking CherryTree data...")
    nodes = walk_tree()
    print(f"Found {len(nodes)} nodes with content")

    documents = []
    metadatas = []
    ids = []

    for node in nodes:
        chunks = chunk_content(node["content"])
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 30:
                continue
            doc_id = f"{node['id']}_{i}" if len(chunks) > 1 else node["id"]
            documents.append(chunk)
            metadatas.append({
                "name": node["name"],
                "path": node["path"],
                "node_id": node["id"],
            })
            ids.append(doc_id)

    print(f"Split into {len(documents)} chunks")

    print("Connecting to Ollama for embeddings...")
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    except requests.ConnectionError:
        print(f"Error: Cannot reach Ollama at {OLLAMA_URL}")
        print("Make sure Ollama is running: ollama serve")
        return

    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name="cherrytree_notes",
        embedding_function=OllamaEmbeddingFunction(EMBED_MODEL, OLLAMA_URL),
    )

    existing = collection.get(ids=ids[:100])
    existing_ids = set(existing["ids"]) if existing["ids"] else set()

    new_docs = []
    new_metas = []
    new_ids = []
    for doc, meta, did in zip(documents, metadatas, ids):
        if did not in existing_ids:
            new_docs.append(doc)
            new_metas.append(meta)
            new_ids.append(did)

    if not new_docs:
        print("All documents already indexed. Nothing to do.")
        return

    print(f"Indexing {len(new_docs)} new chunks in batches...")
    batch_size = 10
    for i in tqdm(range(0, len(new_docs), batch_size)):
        batch_end = min(i + batch_size, len(new_docs))
        collection.add(
            documents=new_docs[i:batch_end],
            metadatas=new_metas[i:batch_end],
            ids=new_ids[i:batch_end],
        )

    print(f"Done! {collection.count()} total chunks in the database.")


if __name__ == "__main__":
    main()
