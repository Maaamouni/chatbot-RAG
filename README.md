# Private RAG Chatbot — CherryTree Notes

A private, offline chatbot that answers your questions based on your personal CherryTree notes. Uses **RAG (Retrieval-Augmented Generation)** — all processing happens locally on your machine with Ollama.

## Pipeline

```
CherryTree XML Notes
        │
        ▼
┌───────────────────┐
│     ingest.py      │  walks the CherryTree directory tree,
│                    │  parses every node.xml, extracts rich_text
│                    │  content, and chunks it into 1000-char pieces
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   nomic-embed-text │  each chunk is embedded into a 768-dim
│   (via Ollama)     │  vector using a dedicated embedding model
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│     ChromaDB       │  stores the embeddings + text + metadata
│  (vector database) │  on disk for fast similarity search
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│     chat.py        │  1. embeds your question (search_query prefix)
│                    │  2. finds top-6 most similar chunks in ChromaDB
│  (RAG query flow)  │  3. builds a prompt: context + question
│                    │  4. sends to qwen2.5:1.5b via Ollama
└────────┬──────────┘
         │
         ▼
    Answer printed to terminal
```

### Why RAG?

- Your notes stay on your machine — **zero data leaves your computer**
- The LLM answers are grounded in your actual notes, not just general knowledge
- Adding new notes only requires re-running `ingest`

## Requirements

- **Ollama** (already installed)  
  `ollama --version` → `0.24.0`

- **Python 3.13+** (already installed)

## Setup

### 1. Install Ollama models

```bash
# Embedding model (for vector search)
ollama pull nomic-embed-text

# Chat model (lightweight & fast on CPU)
ollama pull qwen2.5:1.5b
```

### 2. Install Python dependencies

```bash
python3 -m venv /tmp/chatbot_venv
/tmp/chatbot_venv/bin/pip install -r requirements.txt
```

Or use the auto-setup via `run.sh` (creates the venv automatically on first run).

## Commands

### Index your notes

```bash
./run.sh ingest
```

Parses all `node.xml` files in `~/Desktop/outhmanes_stuff/`, chunks the text, computes embeddings via Ollama, and stores everything in `chroma_db/`.

### Ask a question (one-shot)

```bash
./run.sh chat "What does my notes say about RBAC?"
./run.sh chat "How do I configure Docker?"
```

The answer will be printed directly in the terminal.

### Interactive chat mode

```bash
./run.sh chat
```

Type your questions one by one. The conversation history is kept for follow-up questions.

```
You: what is the CIA triad?
Bot: ...
You: tell me more about integrity
Bot: ...
You: quit
```

### Re-index from scratch

```bash
./run.sh reindex
```

Deletes the old index and rebuilds everything. Do this if you add/change notes in `outhmanes_stuff/`.

## Files

| File | Purpose |
|---|---|
| `ingest.py` | Parses CherryTree XML, chunks text, embeds, indexes into ChromaDB |
| `chat.py` | CLI chatbot: retrieves relevant context + generates answers via Ollama |
| `run.sh` | Convenience wrapper (auto-creates venv, dispatches commands) |
| `requirements.txt` | Python dependencies |
| `chroma_db/` | Persistent vector index (auto-generated, do not edit) |

## How it works

1. **ingest.py** walks the CherryTree directory tree using `subnodes.lst` files to find parent-child relationships, parses each `node.xml` to extract `<rich_text>` content, chunks notes longer than 1000 characters with 100-char overlap, and indexes them into ChromaDB with Ollama embeddings.

2. **chat.py** embeds your question with the `search_query:` prefix (required by nomic-embed-text), retrieves the top-6 most similar chunks from ChromaDB, builds a prompt with the retrieved context, and sends it to `qwen2.5:1.5b` for generation. Streaming responses are printed in real-time.

## Customization

- **Change chat model**: edit `CHAT_MODEL` in `chat.py` (e.g., `"qwen3.5:2b"`, `"codellama:latest"`)
- **Change embedding model**: edit `EMBED_MODEL` in both `ingest.py` and `chat.py`
- **Adjust context size**: change `MAX_CONTEXT_CHARS` in `chat.py`
- **Change number of retrieved chunks**: change `TOP_K` in `chat.py`
- **Change data source**: edit `DATA_DIR` in `ingest.py`
