#!/usr/bin/env python3
"""Build a persistent local ChromaDB index from processed/rag_chunks.json."""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent
CHUNKS_FILE = PROJECT_DIR / "processed" / "rag_chunks.json"
DB_DIR = PROJECT_DIR / "processed" / "chroma_db"
MANIFEST_FILE = PROJECT_DIR / "processed" / "vector_db_manifest.json"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "harrier_manual_chunks"
EMBED_BATCH_SIZE = 16
DB_BATCH_SIZE = 64

REQUIRED_FIELDS = {
    "chunk_id", "source_file", "page_start", "page_end", "section",
    "content", "element_ids", "element_types", "chunk_kind",
}


def load_chunks() -> list[dict[str, Any]]:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Chunk file not found: {CHUNKS_FILE}")

    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("rag_chunks.json must contain a non-empty JSON list.")

    errors = []
    seen_ids = set()
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            errors.append(f"Record {i} is not a JSON object.")
            continue
        missing = REQUIRED_FIELDS - chunk.keys()
        if missing:
            errors.append(f"Record {i} is missing fields: {sorted(missing)}")
            continue
        cid = str(chunk["chunk_id"]).strip()
        if not cid:
            errors.append(f"Record {i} has an empty chunk_id.")
        elif cid in seen_ids:
            errors.append(f"Duplicate chunk_id: {cid}")
        seen_ids.add(cid)
        if not isinstance(chunk["content"], str) or not chunk["content"].strip():
            errors.append(f"Record {i} ({cid}) has empty content.")

    if errors:
        raise ValueError("Chunk validation failed:\n" + "\n".join(errors[:20]))
    return chunks


def fingerprint(chunks: list[dict[str, Any]]) -> str:
    payload = {"model": MODEL_NAME, "chunks": chunks}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_manifest() -> dict[str, Any] | None:
    try:
        with MANIFEST_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def to_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    # Chroma metadata values must be scalar types.
    return {
        "source_file": str(chunk["source_file"]),
        "page_start": int(chunk["page_start"]),
        "page_end": int(chunk["page_end"]),
        "section": str(chunk.get("section") or ""),
        "chunk_kind": str(chunk.get("chunk_kind") or ""),
        "element_types": ", ".join(map(str, chunk.get("element_types", []))),
        "element_ids": ", ".join(map(str, chunk.get("element_ids", []))),
        "table_id": str(chunk.get("table_id") or ""),
        "table_part": int(chunk.get("table_part") or 0),
        "table_parts": int(chunk.get("table_parts") or 0),
    }


def main() -> None:
    print("=== Harrier RAG: Vector Database Builder ===")
    print(f"Project: {PROJECT_DIR}")
    print(f"Chunks : {CHUNKS_FILE}")
    print(f"DB     : {DB_DIR}")
    print(f"Model  : {MODEL_NAME}")

    chunks = load_chunks()
    current_fingerprint = fingerprint(chunks)
    source_docs = sorted({str(c["source_file"]) for c in chunks})
    print(f"Validated chunks: {len(chunks)}")
    print(f"Source documents: {len(source_docs)}")

    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    existing = {c.name for c in client.list_collections()}
    manifest = read_manifest()

    if (manifest
        and manifest.get("fingerprint") == current_fingerprint
        and manifest.get("collection_name") == COLLECTION_NAME
        and COLLECTION_NAME in existing):
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() == len(chunks):
            print(f"Index already current: {collection.count()} vectors. Skipping rebuild.")
            return

    print("Loading embedding model on CPU...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    dimension = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {dimension}")

    # Remove the manifest first so an interrupted rebuild cannot be mistaken
    # for a complete, up-to-date index on the next run.
    if MANIFEST_FILE.exists():
        MANIFEST_FILE.unlink()
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "description": "Local RAG index for Harrier vehicle manuals",
            "embedding_model": MODEL_NAME,
        },
    )

    total = len(chunks)
    for start in range(0, total, DB_BATCH_SIZE):
        batch = chunks[start:start + DB_BATCH_SIZE]
        texts = [str(c["content"]) for c in batch]
        vectors = model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        collection.add(
            ids=[str(c["chunk_id"]) for c in batch],
            documents=texts,
            metadatas=[to_metadata(c) for c in batch],
            embeddings=vectors.tolist(),
        )
        print(f"Indexed {min(start + len(batch), total)}/{total}")

    stored = collection.count()
    if stored != total:
        raise RuntimeError(f"Index check failed: expected {total}, found {stored}")

    manifest_data = {
        "fingerprint": current_fingerprint,
        "model_name": MODEL_NAME,
        "collection_name": COLLECTION_NAME,
        "chunk_count": total,
        "source_documents": source_docs,
        "embedding_dimension": dimension,
        "distance_metric": "cosine",
        "database_directory": str(DB_DIR),
    }
    with MANIFEST_FILE.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, ensure_ascii=False, indent=2)

    print("\n=== VECTOR DATABASE BUILD COMPLETE ===")
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Vectors    : {stored}")
    print(f"Database   : {DB_DIR}")
    print(f"Manifest   : {MANIFEST_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
