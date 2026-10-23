#!/usr/bin/env python3
"""Test local retrieval from the Harrier manuals ChromaDB index.

Examples:
    python3 test_retrieval.py
    python3 test_retrieval.py --query "How do I pair a phone with Bluetooth?"
    python3 test_retrieval.py --query "What does the rotary knob do?" --top-k 5
"""

import argparse
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent
DB_DIR = PROJECT_DIR / "processed" / "chroma_db"
COLLECTION_NAME = "harrier_manual_chunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE recommends a query-side instruction for retrieval tasks.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def parse_args():
    parser = argparse.ArgumentParser(description="Test retrieval from the local Harrier manuals vector database.")
    parser.add_argument("--query", "-q", help="Question to search for. If omitted, enter it interactively.")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of chunks to retrieve (default: 5).")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    query = (args.query or input("Enter your question: ")).strip()
    if not query:
        raise ValueError("Question cannot be empty.")

    if not DB_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found: {DB_DIR}\n"
            "Run build_vector_db.py first."
        )

    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' was not found. "
            "Run build_vector_db.py first."
        ) from exc

    total = collection.count()
    if total == 0:
        raise RuntimeError("The vector collection is empty.")

    print(f"Loaded collection: {COLLECTION_NAME} ({total} vectors)")
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    query_text = QUERY_PREFIX + query
    query_embedding = model.encode(
        query_text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(args.top_k, total),
        include=["documents", "metadatas", "distances"],
    )

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print("\n" + "=" * 88)
    print(f"QUESTION: {query}")
    print("=" * 88)

    for rank, (chunk_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances), start=1
    ):
        # With cosine distance, smaller is better; 1 - distance is a convenient
        # similarity indicator, not a calibrated probability or confidence score.
        similarity = 1.0 - float(distance)
        print(f"\n[{rank}] Chunk ID: {chunk_id}")
        print(f"    Similarity indicator: {similarity:.4f} | Cosine distance: {float(distance):.4f}")
        print(f"    Document: {metadata.get('source_file', 'Unknown')}")
        print(f"    Pages: {metadata.get('page_start', '?')}-{metadata.get('page_end', '?')}")
        print(f"    Section: {metadata.get('section') or '(not specified)'}")
        print(f"    Type: {metadata.get('chunk_kind') or '(not specified)'}")
        if metadata.get("table_id"):
            print(f"    Table ID: {metadata['table_id']}")
        print("    Content:")
        print("-" * 88)
        print(document.strip())
        print("-" * 88)

    print("\nNote: similarity indicators help rank results; they are not answer-confidence percentages.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
