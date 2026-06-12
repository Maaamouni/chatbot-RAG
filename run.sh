#!/usr/bin/env bash
set -e

VENV=/tmp/chatbot_venv
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q chromadb requests tqdm
fi

case "${1:-chat}" in
    ingest)
        echo "=== Ingesting CherryTree data ==="
        "$VENV/bin/python" -u "$SCRIPT_DIR/ingest.py"
        ;;
    chat)
        shift 2>/dev/null || true
        echo "=== Starting Private Chatbot ==="
        echo ""
        "$VENV/bin/python" -u "$SCRIPT_DIR/chat.py" "$@"
        ;;
    reindex)
        echo "=== Re-indexing from scratch ==="
        rm -rf "$SCRIPT_DIR/chroma_db"
        "$VENV/bin/python" -u "$SCRIPT_DIR/ingest.py"
        ;;
    *)
        echo "Usage: $0 [ingest|chat|reindex]"
        echo "  ingest   - Index/update notes from CherryTree"
        echo "  chat     - Start interactive chatbot (default)"
        echo "  reindex  - Delete and re-index all notes"
        exit 1
        ;;
esac
