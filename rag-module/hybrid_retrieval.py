#!/usr/bin/env python3
"""
Hybrid retrieval v2 for the local Harrier manuals RAG project.

Improvements over hybrid_retrieval.py:
- Expands dense and BM25 candidate pools.
- BM25 indexes content plus section and table metadata.
- Adds an entity/section candidate pass so table chunks are not lost just
  because the question includes a term (e.g. anti-clockwise) absent from them.
- Separates direct-answer evidence from related component/table evidence.
- Uses rank fusion and conservative evidence checks.
- Supports explicit manual filtering.
- Can show diagnostic candidates with --debug.

Run:
  python3 hybrid_retrieval_v2.py
  python3 hybrid_retrieval_v2.py --query "What happens if the rotary knob on the button panel is rotated anti-clockwise in ConnectNext?"
  python3 hybrid_retrieval_v2.py --query "What happens if the rotary knob on the button panel is rotated anti-clockwise in ConnectNext?" --debug
"""

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent
CHUNKS_FILE = PROJECT_DIR / "processed" / "rag_chunks.json"
DB_DIR = PROJECT_DIR / "processed" / "chroma_db"
COLLECTION_NAME = "harrier_manual_chunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DENSE_CANDIDATES = 100
BM25_CANDIDATES = 100
RRF_K = 60

# Initial values for evaluation, not accuracy guarantees.
MIN_COVERAGE_FOR_DUAL_CHANNEL = 0.25
STRONG_DENSE_DISTANCE = 0.30
MIN_COVERAGE_FOR_STRONG_DENSE = 0.15
STRONG_BM25_MAX_RANK = 10
DUAL_CHANNEL_MAX_RANK = 40

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "what", "when", "where", "why", "how", "which",
    "who", "whom", "if", "then", "than", "and", "or", "but", "to", "of",
    "in", "on", "at", "by", "for", "from", "with", "without", "into",
    "about", "as", "it", "this", "that", "these", "those", "we", "i",
    "you", "your", "my", "our", "their", "they", "he", "she", "me",
    "can", "could", "would", "should", "will", "may", "might", "please",
    "happens", "happen", "happening", "happened", "get", "gets", "got",
    "use", "using", "used", "tell", "explain", "describe", "manual",
    "harrier", "system", "vehicle", "car", "infotainment",
}
SYNONYMS = {
    "rotated": "rotate", "rotating": "rotate", "rotation": "rotate",
    "rotates": "rotate", "turn": "rotate", "turns": "rotate",
    "turned": "rotate", "turning": "rotate",
    "presses": "press", "pressed": "press", "pressing": "press",
    "favourites": "favourite", "favorite": "favourite",
    "favorites": "favourite", "knobs": "knob", "buttons": "button",
    "functions": "function", "functionality": "function",
    "functionalities": "function",
}


def normalize_text(text):
    text = (text or "").lower()
    text = re.sub(r"anti[\s\-]*clock[\s\-]*wise", "anticlockwise", text)
    text = re.sub(r"long[\s\-]+press", "longpress", text)
    text = re.sub(r"7[\s\-]*inch", "seveninch", text)
    return text


def tokenize(text, remove_stopwords=True):
    words = re.findall(r"[a-z]+|[0-9]+", normalize_text(text))
    result = []
    for word in words:
        word = SYNONYMS.get(word, word)
        if remove_stopwords and word in STOPWORDS:
            continue
        if len(word) == 1 and not word.isdigit():
            continue
        result.append(word)
    return result


def source_filter_for(query):
    q = normalize_text(query)
    if "seveninch" in q or "7-inch" in query.lower() or "7 inch" in query.lower():
        return "harrier-7-inch-infotainment-manual.json"
    if re.search(r"\bmce\b", q):
        return "harrier-mce-infotainment-system-user-manual-feb-2024.json"
    if "owners manual" in q or "owner manual" in q or "bs6" in q:
        return "harrier-bs6-owners-manual-april-2026.json"
    return None


def enriched_text(chunk):
    """Search content and structural metadata; preserve fields for ranking."""
    fields = [
        chunk.get("section", ""),
        chunk.get("chunk_kind", ""),
        chunk.get("source_file", ""),
        chunk.get("content", ""),
        chunk.get("table_id", ""),
    ]
    return " ".join(str(value) for value in fields if value is not None)


def bm25_scores(query_tokens, documents):
    tokenized = [tokenize(enriched_text(d)) for d in documents]
    lengths = [len(tokens) for tokens in tokenized]
    avgdl = sum(lengths) / max(len(lengths), 1)
    doc_freq = Counter()
    term_freqs = []
    for tokens in tokenized:
        tf = Counter(tokens)
        term_freqs.append(tf)
        doc_freq.update(set(tokens))

    n = len(documents)
    k1, b = 1.5, 0.75
    scores = []
    for tf, length in zip(term_freqs, lengths):
        score = 0.0
        for term in query_tokens:
            freq = tf.get(term, 0)
            if not freq:
                continue
            df = doc_freq[term]
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            denom = freq + k1 * (1 - b + b * length / max(avgdl, 1))
            score += idf * freq * (k1 + 1) / denom
        scores.append(score)
    return scores


def coverage(query_tokens, chunk):
    # Include section/type labels because they are useful for component/table queries.
    terms = set(query_tokens)
    doc_terms = set(tokenize(enriched_text(chunk)))
    return len(terms & doc_terms) / max(len(terms), 1)


def explicit_entity_terms(query_tokens):
    """Return high-value component/action tokens for structural candidate expansion."""
    preferred = {
        "rotary", "knob", "button", "panel", "favourite", "volume",
        "seek", "home", "clockwise", "anticlockwise", "rotate", "press",
        "longpress", "radio", "frequency", "connectnext",
    }
    return [t for t in query_tokens if t in preferred]


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid retrieval v2 (dense + BM25 + table-aware candidate expansion).")
    parser.add_argument("--query", "-q", help="Question to search for.")
    parser.add_argument("--top-k", "-k", type=int, default=5)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")
    query = (args.query or input("Enter your question: ")).strip()
    if not query:
        raise ValueError("Question cannot be empty.")
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Missing chunk file: {CHUNKS_FILE}")
    if not DB_DIR.exists():
        raise FileNotFoundError(f"Missing ChromaDB directory: {DB_DIR}")

    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        chunks = json.load(f)
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("rag_chunks.json is empty or invalid.")

    by_id = {str(c["chunk_id"]): c for c in chunks}
    source_filter = source_filter_for(query)
    corpus = chunks
    if source_filter:
        corpus = [c for c in chunks if str(c.get("source_file", "")).lower() == source_filter.lower()]
        if not corpus:
            raise RuntimeError(f"No chunks found for explicitly requested manual: {source_filter}")

    query_tokens = list(dict.fromkeys(tokenize(query)))
    if not query_tokens:
        raise ValueError("No searchable terms remain after normalization.")
    entities = explicit_entity_terms(query_tokens)

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    total = collection.count()
    if total == 0:
        raise RuntimeError("ChromaDB collection is empty.")

    print(f"Collection: {COLLECTION_NAME} ({total} vectors)")
    if source_filter:
        print(f"Document boundary: {source_filter}")
    print(f"Query terms: {', '.join(query_tokens)}")
    print(f"Entity terms used for candidate expansion: {', '.join(entities) or '(none)'}")

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    qvec = model.encode(QUERY_PREFIX + query, convert_to_numpy=True, normalize_embeddings=True)

    where = {"source_file": source_filter} if source_filter else None
    dense = collection.query(
        query_embeddings=[qvec.tolist()],
        n_results=min(DENSE_CANDIDATES, total),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    dense_rank, dense_distance = {}, {}
    for rank, (cid, dist) in enumerate(zip(dense["ids"][0], dense["distances"][0]), start=1):
        dense_rank[cid] = rank
        dense_distance[cid] = float(dist)

    scores = bm25_scores(query_tokens, corpus)
    bm25_order = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)
    bm25_rank, bm25_score_by_id = {}, {}
    for rank, idx in enumerate(bm25_order, start=1):
        cid = str(corpus[idx]["chunk_id"])
        bm25_rank[cid] = rank
        bm25_score_by_id[cid] = scores[idx]

    candidates = set(dense_rank)
    candidates.update(
        str(corpus[i]["chunk_id"])
        for i in bm25_order[:min(BM25_CANDIDATES, len(bm25_order))]
        if scores[i] > 0
    )

    # Structural/entity expansion: include chunks whose section or content matches
    # multiple component terms, especially table chunks, even if they missed top-N.
    expanded_ids = set()
    for chunk in corpus:
        text = enriched_text(chunk)
        terms = set(tokenize(text))
        overlap = len(set(entities) & terms)
        section = normalize_text(str(chunk.get("section", "")))
        is_table = str(chunk.get("chunk_kind", "")).lower() == "table"
        if overlap >= 2 and (is_table or "button panel" in section or "control elements" in section):
            cid = str(chunk["chunk_id"])
            candidates.add(cid)
            expanded_ids.add(cid)

    fused = []
    for cid in candidates:
        chunk = by_id.get(cid)
        if not chunk:
            continue
        dr = dense_rank.get(cid)
        br = bm25_rank.get(cid)
        dist = dense_distance.get(cid)
        cov = coverage(query_tokens, chunk)
        terms = set(tokenize(enriched_text(chunk)))
        entity_overlap = len(set(entities) & terms)
        is_table = str(chunk.get("chunk_kind", "")).lower() == "table"
        section = normalize_text(str(chunk.get("section", "")))

        # Weighted RRF. Keyword channel gets a small boost.
        rrf = (1 / (RRF_K + dr) if dr else 0) + (1.15 / (RRF_K + br) if br else 0)

        # Explicit entity overlap and relevant table structure are ranking aids,
        # not substitutes for answer evidence.
        entity_bonus = min(entity_overlap, 4) * 0.0009
        table_bonus = 0.0015 if is_table and entity_overlap >= 2 else 0.0
        button_panel_bonus = 0.0010 if "button panel" in section and entity_overlap >= 2 else 0.0
        final_score = rrf + entity_bonus + table_bonus + button_panel_bonus

        strong_bm25 = br is not None and br <= STRONG_BM25_MAX_RANK and cov >= MIN_COVERAGE_FOR_DUAL_CHANNEL
        dual_channel = (
            dr is not None and br is not None
            and dr <= DUAL_CHANNEL_MAX_RANK and br <= DUAL_CHANNEL_MAX_RANK
            and cov >= MIN_COVERAGE_FOR_DUAL_CHANNEL
        )
        strong_dense = (
            dr is not None and dist is not None
            and dist <= STRONG_DENSE_DISTANCE
            and cov >= MIN_COVERAGE_FOR_STRONG_DENSE
        )
        # Related table/component evidence may be returned in a separate evidence
        # category when component overlap is strong. It is not labelled as a direct answer.
        related_table = is_table and entity_overlap >= 2
        accepted = strong_bm25 or dual_channel or strong_dense or related_table

        reasons = []
        if strong_bm25:
            reasons.append("strong BM25 + coverage")
        if dual_channel:
            reasons.append("dense/BM25 agreement + coverage")
        if strong_dense:
            reasons.append("strong dense + coverage")
        if related_table:
            reasons.append("related table/component evidence")
        if not reasons:
            reasons.append("rejected: insufficient evidence")

        # A chunk containing the action term is more likely direct evidence;
        # related component tables are deliberately separated in output.
        action_terms = {"anticlockwise", "clockwise", "rotate", "frequency", "decrease", "increase"}
        action_overlap = len(action_terms & terms)
        category = "direct/likely answer" if action_overlap >= 2 else (
            "related component/table evidence" if related_table else "general context"
        )
        fused.append({
            "id": cid, "chunk": chunk, "dense_rank": dr, "distance": dist,
            "bm25_rank": br, "bm25_score": bm25_score_by_id.get(cid, 0.0),
            "coverage": cov, "entity_overlap": entity_overlap, "score": final_score,
            "accepted": accepted, "reason": "; ".join(reasons),
            "category": category, "expanded": cid in expanded_ids,
        })

    fused.sort(key=lambda x: x["score"], reverse=True)
    accepted = [x for x in fused if x["accepted"]]
    direct = [x for x in accepted if x["category"] == "direct/likely answer"]
    related = [x for x in accepted if x["category"] != "direct/likely answer"]

    print("\\n" + "=" * 92)
    print(f"QUESTION: {query}")
    print("=" * 92)

    def print_item(rank, item):
        c = item["chunk"]
        print(f"\\n[{rank}] {item['category']} | ID: {item['id']}")
        print(f"    Source: {c.get('source_file')}")
        print(f"    Pages: {c.get('page_start')}-{c.get('page_end')} | Section: {c.get('section')} | Type: {c.get('chunk_kind')}")
        print(f"    Dense rank/distance: {item['dense_rank']} / {item['distance']}")
        print(f"    BM25 rank/score: {item['bm25_rank']} / {item['bm25_score']:.4f}")
        print(f"    Query-term coverage: {item['coverage']:.0%} | Entity overlap: {item['entity_overlap']}")
        if item["expanded"]:
            print("    Candidate source: structural/entity expansion")
        print(f"    Guardrail: ACCEPT ({item['reason']})")
        print("    Content:")
        print("-" * 92)
        print(c.get("content", "").strip())
        print("-" * 92)

    if not accepted:
        print("NO RESULTS PASSED THE RELEVANCE GUARDRAILS.")
        print("This means evidence was insufficient under current rules; it does not prove the manual lacks an answer.")
    else:
        # Direct answer evidence first, then related component/table evidence.
        ordered = direct + related
        for i, item in enumerate(ordered[:args.top_k], start=1):
            print_item(i, item)

    if args.debug:
        print("\\n--- REJECTED CANDIDATES (diagnostics only) ---")
        rejected = [x for x in fused if not x["accepted"]]
        for item in rejected[:15]:
            c = item["chunk"]
            print(
                f"ID={item['id']} | source={c.get('source_file')} | "
                f"pages={c.get('page_start')}-{c.get('page_end')} | "
                f"type={c.get('chunk_kind')} | dense={item['dense_rank']}/{item['distance']} | "
                f"BM25={item['bm25_rank']} | coverage={item['coverage']:.0%} | "
                f"entities={item['entity_overlap']} | {item['reason']}"
            )
        print("\\n--- EXPANDED TABLE CANDIDATES ---")
        for item in fused:
            if item["expanded"]:
                c = item["chunk"]
                print(f"{item['id']} | pages={c.get('page_start')}-{c.get('page_end')} | section={c.get('section')} | accepted={item['accepted']}")

    print("\\nImportant: these are retrieval candidates, not verified answers. Check the source text and page references.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
