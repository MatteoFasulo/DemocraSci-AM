#!/usr/bin/env bash
set -u

# A Renku App has one externally routed port. Keep FastAPI private inside the
# same container and expose only Streamlit through the Renku-assigned port.
api_port="${DEMOCRASCI_API_PORT:-8000}"
ui_port="${RENKU_SESSION_PORT:-8501}"

# Chroma uses SQLite and local index files, so download the bucket contents to
# the App's local filesystem instead of opening the database over object storage.
bucket_id="${HF_CHROMA_BUCKET:-MatteoFasulo/DemocraSci-ChromaDB}"
cache_dir="${CHROMA_CACHE_DIR:-/tmp/democrasci-chromadb}"
# Renku sessions can mount saved secrets as files. Renku Apps cannot access
# session secrets, so this is optional and must never be configured as an App
# launcher variable.
hf_token_file="${HF_TOKEN_FILE:-/secrets/hf_token.txt}"
if [[ -z "${HF_TOKEN:-}" && -r "$hf_token_file" ]]; then
  HF_TOKEN="$(tr -d '\r\n' < "$hf_token_file")"
  export HF_TOKEN
fi

if [[ -n "${CHROMA_PATH:-}" && -f "${CHROMA_PATH}/chroma.sqlite3" ]]; then
  chroma_path="$CHROMA_PATH"
elif cached_sqlite="$(find "$cache_dir" -type f -name chroma.sqlite3 -print -quit 2>/dev/null)" && [[ -n "$cached_sqlite" ]]; then
  chroma_path="$(dirname "$cached_sqlite")"
else
  if ! command -v hf >/dev/null 2>&1; then
    echo "ERROR: Hugging Face CLI 'hf' is missing; install huggingface-hub >= 1.0.0." >&2
    exit 1
  fi

  mkdir -p "$(dirname "$cache_dir")"
  staging_dir="$(mktemp -d "${cache_dir}.download.XXXXXX")"
  echo "Downloading Chroma index from Hugging Face bucket ${bucket_id}..."
  if ! hf buckets sync "hf://buckets/${bucket_id}" "$staging_dir"; then
    rm -rf "$staging_dir"
    echo "ERROR: Could not download bucket ${bucket_id}. Confirm it is public, or that this runtime has valid Hugging Face credentials." >&2
    exit 1
  fi

  sqlite_file="$(find "$staging_dir" -type f -name chroma.sqlite3 -print -quit)"
  if [[ -z "$sqlite_file" ]]; then
    rm -rf "$staging_dir"
    echo "ERROR: No chroma.sqlite3 found in bucket ${bucket_id}; upload the full Chroma persistence directory." >&2
    exit 1
  fi
  relative_db_dir="$(dirname "${sqlite_file#"$staging_dir"/}")"
  if [[ ! -e "$cache_dir" ]]; then
    mv "$staging_dir" "$cache_dir"
    if [[ "$relative_db_dir" == "." ]]; then
      chroma_path="$cache_dir"
    else
      chroma_path="$cache_dir/$relative_db_dir"
    fi
  else
    chroma_path="$(dirname "$sqlite_file")"
  fi
  echo "Chroma index downloaded to ${chroma_path}"
fi

export CHROMA_PATH="$chroma_path"

uvicorn model_service:app --host 127.0.0.1 --port "$api_port" &
api_pid=$!

streamlit run dashboard.py \
  --server.port "$ui_port" \
  --server.address 0.0.0.0 \
  --server.headless true &
ui_pid=$!

cleanup() {
  kill "$api_pid" "$ui_pid" 2>/dev/null || true
  wait "$api_pid" "$ui_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Stop the container if either service exits.
wait -n "$api_pid" "$ui_pid"
