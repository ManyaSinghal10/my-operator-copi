#!/usr/bin/env python3

"""Improved local hybrid retrieval API for the Harrier manuals.

Features:

- Dense retrieval + BM25 + reciprocal-rank fusion.

- Question-to-passage lexical normalization and retrieval-only synonyms.

- Section-aware scoring for headings such as VOICE COMMAND GUIDELINES.

- Table-aware scoring and same-table expansion.

- Conservative evidence acceptance / abstention.

- Returns evidence and metadata only; the chat client handles answer generation.

Run from the project root:

    python3 kb_server.py

Endpoints:

    GET  /health

    GET  /documents

    POST /retrieve

"""

import json

import hashlib
import math
import pickle

import re

import sys

from collections import Counter, defaultdict

from contextlib import asynccontextmanager

from pathlib import Path

from typing import Optional

import chromadb

import uvicorn

from fastapi import FastAPI, HTTPException

from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent

CHUNKS_FILE = PROJECT_DIR / "processed" / "rag_chunks.json"
BM25_CACHE_FILE = PROJECT_DIR / "processed" / "bm25_index.pkl"
BM25_CACHE_VERSION = 1

DB_DIR = PROJECT_DIR / "processed" / "chroma_db"

COLLECTION_NAME = "harrier_manual_chunks"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

HOST, PORT = "127.0.0.1", 8001

DENSE_CANDIDATES = 100

BM25_CANDIDATES = 150

RRF_K = 60

DEFAULT_TOP_K = 5

STOPWORDS = set("""

a an the is are was were be been being do does did what when where why how

which who whom if then than and or but to of in on at by for from with

without into about as it this that these those we i you your my our their

they he she me can could would should will may might please happens happen

happening happened get gets got use using used tell explain describe manual

harrier system vehicle car infotainment feature features following following

from available provide provides tell me show list

""".split())

# Retrieval-only normalization/synonyms. These do NOT alter returned manual text.

SYNONYMS = {

    "rotated": "rotate", "rotating": "rotate", "rotation": "rotate",

    "rotates": "rotate", "turn": "rotate", "turns": "rotate",

    "turned": "rotate", "turning": "rotate",

    "presses": "press", "pressed": "press", "pressing": "press",

    "longpress": "longpress", "longpressed": "longpress",

    "guidelines": "guideline", "commands": "command", "commanding": "command",

    "favourites": "favourite", "favorite": "favourite", "favorites": "favourite",

    "knobs": "knob", "buttons": "button", "functions": "function",

    "functionality": "function", "functionalities": "function",

    "settings": "setting", "controls": "control", "controlled": "control",

    "controlling": "control", "recognition": "recognition",

    "voice": "voice", "microphone": "voice",

    "clockwise": "clockwise", "anticlockwise": "anticlockwise",

    "anti-clockwise": "anticlockwise", "counterclockwise": "anticlockwise",

    "decrease": "decrease", "decreases": "decrease",

    "increases": "increase", "increasing": "increase",

    "increased": "increase", "decreased": "decrease",

}

# Short, retrieval-only query expansions for common manual phrasing.

QUERY_EXPANSIONS = {

    "voice": {"voice", "voice command", "voice recognition", "speech"},

    "guideline": {"guideline", "guidelines", "speak", "speaking", "accent", "noise", "pause"},

    "longpress": {"long press", "long-press", "long press operation"},

    "press": {"press", "on knob press", "short press", "long press"},

    "button": {"button", "button panel", "button-panel", "control"},

    "setting": {"setting", "setup", "configuration"},

    "rotary": {"rotary", "knob", "rotary knob"},

    "climate": {"climate", "hvac", "temperature"},

    "radio": {"radio", "fm", "am", "frequency", "station"},

}

STATE = {}

def normalize_text(text: str) -> str:

    text = (text or "").lower()

    text = re.sub(r"anti[\s-]*clock[\s-]*wise", "anticlockwise", text)

    text = re.sub(r"counter[\s-]*clock[\s-]*wise", "anticlockwise", text)

    text = re.sub(r"long[\s-]+press(?:ed|ing)?", "longpress", text)

    text = re.sub(r"7[\s-]*inch", "seveninch", text)

    text = re.sub(r"voice[\s-]+recognition", "voice recognition", text)

    text = re.sub(r"[\u2010-\u2015]", "-", text)

    return text

def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:

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

def expanded_query_tokens(question: str) -> list[str]:

    base = tokenize(question)

    expanded = list(base)

    for term in base:

        for synonym in QUERY_EXPANSIONS.get(term, set()):

            expanded.extend(tokenize(synonym))

    # Keep stable order and avoid letting expansions overwhelm exact terms.

    return list(dict.fromkeys(expanded))

def source_filter_for(query: str) -> Optional[str]:

    q = normalize_text(query)

    if "seveninch" in q or "7-inch" in query.lower() or "7 inch" in query.lower():

        return "harrier-7-inch-infotainment-manual.json"

    if re.search(r"\bmce\b", q):

        return "harrier-mce-infotainment-system-user-manual-feb-2024.json"

    if "owners manual" in q or "owner manual" in q or "bs6" in q:

        return "harrier-bs6-owners-manual-april-2026.json"

    return None

def enriched_text(chunk: dict) -> str:

    return " ".join(str(chunk.get(key, "")) for key in (

        "section", "chunk_kind", "source_file", "content", "table_id"

    ))

class BM25Index:
    """Precomputed BM25 statistics for the full corpus and each source file.

    Tokenization and term/document-frequency calculations happen once when the
    index is built. Query-time scoring reuses the cached token and corpus stats.
    """

    def __init__(self, documents: list[dict]):
        self.tokenized_docs = [tokenize(enriched_text(doc)) for doc in documents]
        self.term_freqs = [Counter(tokens) for tokens in self.tokenized_docs]
        self.lengths = [len(tokens) for tokens in self.tokenized_docs]

        self.scope_indices = {"__all__": list(range(len(documents)))}
        for index, document in enumerate(documents):
            source = str(document.get("source_file", ""))
            self.scope_indices.setdefault(source, []).append(index)

        self.scope_stats = {}
        for scope, indices in self.scope_indices.items():
            document_frequency = Counter()
            for index in indices:
                document_frequency.update(set(self.tokenized_docs[index]))

            self.scope_stats[scope] = {
                "n": len(indices),
                "avgdl": (
                    sum(self.lengths[index] for index in indices) / len(indices)
                    if indices else 0.0
                ),
                "df": document_frequency,
            }

    def scores(self, query_tokens: list[str], source_file: Optional[str] = None) -> list[float]:
        """Return scores aligned with the selected corpus's original chunk order."""
        scope = source_file or "__all__"
        if scope not in self.scope_indices or scope not in self.scope_stats:
            return []

        indices = self.scope_indices[scope]
        stats = self.scope_stats[scope]
        n = stats["n"]
        avgdl = stats["avgdl"]
        df = stats["df"]
        k1, b = 1.5, 0.75
        scores = []

        for index in indices:
            tf = self.term_freqs[index]
            length = self.lengths[index]
            score = 0.0

            for term in query_tokens:
                freq = tf.get(term, 0)
                if not freq:
                    continue

                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                denominator = freq + k1 * (1 - b + b * length / max(avgdl, 1.0))
                score += idf * freq * (k1 + 1) / denominator

            scores.append(score)

        return scores

    def to_cache_data(self) -> dict:
        return {
            "tokenized_docs": self.tokenized_docs,
            "term_freqs": self.term_freqs,
            "lengths": self.lengths,
            "scope_indices": self.scope_indices,
            "scope_stats": self.scope_stats,
        }

    @classmethod
    def from_cache_data(cls, data: dict) -> "BM25Index":
        instance = cls.__new__(cls)
        instance.tokenized_docs = data["tokenized_docs"]
        instance.term_freqs = data["term_freqs"]
        instance.lengths = data["lengths"]
        instance.scope_indices = data["scope_indices"]
        instance.scope_stats = data["scope_stats"]
        return instance


def load_or_build_bm25_index(chunks: list[dict], raw_chunks: bytes) -> tuple[BM25Index, bool]:
    """Load the persistent BM25 cache, rebuilding it if the chunks changed."""
    signature = hashlib.sha256(
        str(BM25_CACHE_VERSION).encode("ascii") + b":" + raw_chunks
    ).hexdigest()

    if BM25_CACHE_FILE.exists():
        try:
            with BM25_CACHE_FILE.open("rb") as cache_file:
                cache = pickle.load(cache_file)
            if cache.get("signature") == signature:
                return BM25Index.from_cache_data(cache["index_data"]), True
        except Exception as exc:
            print(f"BM25 cache could not be loaded; rebuilding it ({exc}).", file=sys.stderr)

    index = BM25Index(chunks)
    BM25_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = BM25_CACHE_FILE.with_suffix(BM25_CACHE_FILE.suffix + ".tmp")
    try:
        with temporary_file.open("wb") as cache_file:
            pickle.dump(
                {"signature": signature, "index_data": index.to_cache_data()},
                cache_file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        temporary_file.replace(BM25_CACHE_FILE)
    except Exception:
        try:
            temporary_file.unlink(missing_ok=True)
        except Exception:
            pass
        raise

    return index, False

def section_intents(question: str) -> list[str]:

    q = normalize_text(question)

    intents = []

    patterns = [

        (r"\bvoice\b|\bspeech\b|\brecognition\b|\bcommand\b|\bguideline\b", ("voice command", "voice recognition", "voice command guidelines")),

        (r"\bsetting\b|\bsetup\b|\bconfigure\b|\bconfiguration\b", ("settings", "overview", "setup")),

        (r"\blongpress\b|\bpress\b|\bbutton\b|\bknob\b|\brotary\b", ("button panel", "button panel controller", "control elements")),

        (r"\brotate\b|\bclockwise\b|\banticlockwise\b|\bradio\b|\bfm\b|\bam\b", ("manual tuning", "button panel", "radio")),

        (r"\bclimate\b|\bhvac\b|\btemperature\b|\bac\b", ("climate control", "using hvac control elements")),

        (r"\bphone\b|\bcall\b|\bcontact\b|\bbluetooth\b", ("phone", "connectivity", "bluetooth")),

        (r"\bmedia\b|\busb\b|\btrack\b|\bmusic\b|\baudio\b", ("audio", "media", "button panel")),

    ]

    for pattern, sections in patterns:

        if re.search(pattern, q):

            intents.extend(sections)

    return list(dict.fromkeys(s.lower() for s in intents))

def is_table_chunk(chunk: dict) -> bool:

    kind = str(chunk.get("chunk_kind", "")).lower()

    content = str(chunk.get("content", "")).lower()

    return (

        "table" in kind

        or bool(chunk.get("table_id"))

        or "table caption:" in content

        or "columns:" in content

    )



def build_retrieval_indexes(chunks: list[dict], bm25_index: BM25Index) -> dict:
    """Build reusable lookup structures so query-time expansion avoids full scans.

    Reuse BM25 cached token lists instead of duplicating token sets for every chunk.
    """
    chunks_by_source = defaultdict(list)
    chunk_position_by_id = {}
    is_table_by_id = {}
    section_ids_by_scope = {"__all__": defaultdict(list)}
    table_ids_by_scope = {"__all__": defaultdict(list)}
    table_token_ids_by_scope = {"__all__": defaultdict(set)}

    for position, chunk in enumerate(chunks):
        cid = str(chunk["chunk_id"])
        source = str(chunk.get("source_file", ""))
        chunks_by_source[source].append(chunk)
        chunk_position_by_id[cid] = position

        # BM25 already stores the tokenized documents; reuse those tokens.
        terms = set(bm25_index.tokenized_docs[position])
        table_flag = is_table_chunk(chunk)
        is_table_by_id[cid] = table_flag

        section = normalize_text(str(chunk.get("section", ""))).strip()
        if section:
            section_ids_by_scope["__all__"][section].append(cid)

        table_id = chunk.get("table_id")
        if table_id is not None:
            table_ids_by_scope["__all__"][(source, str(table_id))].append(cid)

        if table_flag:
            for token in terms:
                table_token_ids_by_scope["__all__"][token].add(cid)

        if source not in section_ids_by_scope:
            section_ids_by_scope[source] = defaultdict(list)
            table_ids_by_scope[source] = defaultdict(list)
            table_token_ids_by_scope[source] = defaultdict(set)

        if section:
            section_ids_by_scope[source][section].append(cid)
        if table_id is not None:
            table_ids_by_scope[source][str(table_id)].append(cid)
        if table_flag:
            for token in terms:
                table_token_ids_by_scope[source][token].add(cid)

    return {
        "chunks_by_source": dict(chunks_by_source),
        "chunk_position_by_id": chunk_position_by_id,
        "is_table_by_id": is_table_by_id,
        "section_ids_by_scope": {scope: dict(index) for scope, index in section_ids_by_scope.items()},
        "table_ids_by_scope": {scope: dict(index) for scope, index in table_ids_by_scope.items()},
        "table_token_ids_by_scope": {scope: dict(index) for scope, index in table_token_ids_by_scope.items()},
    }


def phrase_hits(question: str, chunk: dict) -> int:

    q = normalize_text(question)

    content = normalize_text(enriched_text(chunk))

    phrases = [

        "voice command guidelines", "voice command guideline", "voice recognition",

        "button panel", "rotary knob", "longpress", "long press",

        "manual tuning", "vehicle setup", "audio setup", "phone setup",

        "volume setup", "system setup", "driver assistance", "software update",

    ]

    return sum(1 for phrase in phrases if phrase in q and phrase in content)

def keyword_coverage(question_tokens: list[str], chunk: dict) -> float:

    terms = set(question_tokens)

    passage_terms = set(tokenize(enriched_text(chunk)))

    # Exact question terms are used here, not expansion terms.

    return len(terms & passage_terms) / max(len(terms), 1)

def section_match(question: str, chunk: dict, intents: list[str]) -> tuple[float, list[str]]:

    section = normalize_text(str(chunk.get("section", "")))

    content = normalize_text(str(chunk.get("content", "")))

    reasons = []

    score = 0.0

    if section:

        for intent in intents:

            intent_norm = normalize_text(intent)

            if intent_norm and (intent_norm in section or section in intent_norm):

                score = max(score, 0.006)

                reasons.append("section heading matches question intent")

                break

        q_tokens = set(tokenize(question))

        s_tokens = set(tokenize(section))

        overlap = len(q_tokens & s_tokens)

        if overlap >= 2:

            score += min(overlap, 4) * 0.0018

            reasons.append("question terms overlap section heading")

    # Explicit phrase matches in passage/heading are useful for exact manual terminology.

    hits = phrase_hits(question, chunk)

    if hits:

        score += min(hits, 3) * 0.002

        reasons.append("exact phrase match")

    return score, reasons

def retrieve(question: str, top_k: int = DEFAULT_TOP_K, source_file: Optional[str] = None) -> dict:

    if not STATE.get("ready"):

        raise RuntimeError("Retrieval service is not initialized.")

    chunks, by_id = STATE["chunks"], STATE["by_id"]

    collection, model = STATE["collection"], STATE["model"]

    source = source_file or source_filter_for(question)

    if source and source not in STATE["source_documents"]:

        raise ValueError(f"Unknown source_file '{source}'. Available: {STATE['source_documents']}")

    # Use prebuilt source lookup instead of filtering every chunk per request.
    corpus = chunks if not source else STATE["chunks_by_source"].get(source, [])

    if not corpus:

        raise ValueError(f"No chunks found for source_file '{source}'.")

    exact_qtokens = list(dict.fromkeys(tokenize(question)))

    retrieval_qtokens = expanded_query_tokens(question)

    if not exact_qtokens:

        raise ValueError("Question has no searchable terms after normalization.")

    qvec = model.encode(

        QUERY_PREFIX + question, convert_to_numpy=True, normalize_embeddings=True

    )

    where = {"source_file": source} if source else None

    dense = collection.query(

        query_embeddings=[qvec.tolist()],

        n_results=min(DENSE_CANDIDATES, len(corpus), collection.count()),

        where=where,

        include=["documents", "metadatas", "distances"],

    )

    dense_rank, distances = {}, {}

    for rank, (cid, dist) in enumerate(zip(dense["ids"][0], dense["distances"][0]), 1):

        dense_rank[cid], distances[cid] = rank, float(dist)

    bm25_index = STATE["bm25_index"]
    lexical = bm25_index.scores(retrieval_qtokens, source)

    order = sorted(range(len(corpus)), key=lambda i: lexical[i], reverse=True)

    bmrank, bmscore = {}, {}

    for rank, i in enumerate(order, 1):

        cid = str(corpus[i]["chunk_id"])

        bmrank[cid], bmscore[cid] = rank, lexical[i]

    intents = section_intents(question)

    exact_terms = set(exact_qtokens)
    expanded_terms = set(retrieval_qtokens)

    # Gather candidates from dense and lexical retrieval.

    candidate_ids = set(dense_rank)

    candidate_ids.update(

        str(corpus[i]["chunk_id"])

        for i in order[:min(BM25_CANDIDATES, len(order))]

        if lexical[i] > 0

    )

    # Section-aware expansion uses a section-name index, so it visits distinct
    # section labels rather than scanning every chunk in the corpus.
    section_expanded = set()
    scope = source or "__all__"
    section_index = STATE["section_ids_by_scope"].get(scope, {})
    chunk_position_by_id = STATE["chunk_position_by_id"]
    bm25_index = STATE["bm25_index"]

    for sec, section_chunk_ids in section_index.items():
        heading_match = any(
            normalize_text(intent) in sec or sec in normalize_text(intent)
            for intent in intents if sec
        )
        if not heading_match:
            continue

        for cid in section_chunk_ids:
            position = chunk_position_by_id.get(cid)
            passage_terms = set(bm25_index.tokenized_docs[position]) if position is not None else set()
            if len(exact_terms & passage_terms) >= 1:
                candidate_ids.add(cid)
                section_expanded.add(cid)

    # Table-aware expansion uses token and table-id indexes instead of scanning
    # every chunk repeatedly.
    table_expanded = set()
    control_query = bool(re.search(
        r"\b(button|knob|rotary|press|long[\s-]*press|rotate|clockwise|track|volume|control)\b",
        normalize_text(question),
    ))

    table_seed_ids = set()
    table_threshold = 1 if control_query else 2
    table_token_index = STATE["table_token_ids_by_scope"].get(scope, {})
    possible_table_ids = set()
    for term in exact_terms:
        possible_table_ids.update(table_token_index.get(term, set()))

    for cid in possible_table_ids:
        position = chunk_position_by_id.get(cid)
        terms = set(bm25_index.tokenized_docs[position]) if position is not None else set()
        if len(exact_terms & terms) >= table_threshold:
            candidate_ids.add(cid)
            table_seed_ids.add(cid)
            table_expanded.add(cid)

    table_index = STATE["table_ids_by_scope"].get(scope, {})
    for cid in table_seed_ids:
        chunk = by_id.get(cid)
        if not chunk or chunk.get("table_id") is None:
            continue

        if source:
            sibling_ids = table_index.get(str(chunk.get("table_id")), [])
        else:
            sibling_ids = table_index.get(
                (str(chunk.get("source_file", "")), str(chunk.get("table_id"))), []
            )

        for sibling_id in sibling_ids:
            candidate_ids.add(sibling_id)
            table_expanded.add(sibling_id)

    candidates = []

    for cid in candidate_ids:

        chunk = by_id.get(cid)

        if not chunk or (source and chunk.get("source_file") != source):

            continue

        dr, br, dist = dense_rank.get(cid), bmrank.get(cid), distances.get(cid)

        # Reuse precomputed token sets for candidate overlap calculations.
        position = chunk_position_by_id.get(cid)
        passage_terms = set(bm25_index.tokenized_docs[position]) if position is not None else set()
        # Preserve the previous keyword_coverage calculation exactly for this
        # optimization-only step; a separate step can correct its semantics.
        coverage_terms = set(question)
        cov = len(coverage_terms & passage_terms) / max(len(coverage_terms), 1)

        exact_overlap = len(exact_terms & passage_terms)

        expanded_overlap = len(expanded_terms & passage_terms)

        table = STATE["is_table_by_id"].get(cid, False)

        sec_score, sec_reasons = section_match(question, chunk, intents)

        table_score = 0.0

        reasons = list(sec_reasons)

        score = (1 / (RRF_K + dr) if dr else 0) + (1.15 / (RRF_K + br) if br else 0)

        score += min(exact_overlap, 5) * 0.0015

        score += min(expanded_overlap, 5) * 0.0005

        score += sec_score

        if table and control_query and exact_overlap >= 1:

            table_score += 0.005

            reasons.append("table-aware control/action match")

        if table and exact_overlap >= 2:

            table_score += 0.002

            reasons.append("table contains multiple question terms")

        if cid in table_expanded:

            table_score += 0.001

            reasons.append("table/sibling expansion")

        score += table_score

        strong_bm25 = br is not None and br <= 12 and (cov >= 0.18 or exact_overlap >= 2)

        dual = dr is not None and br is not None and dr <= 45 and br <= 45 and (cov >= 0.18 or exact_overlap >= 2)

        strong_dense = dr is not None and dist is not None and dist <= 0.28 and (cov >= 0.10 or sec_score > 0)

        section_supported = cid in section_expanded and (cov >= 0.10 or exact_overlap >= 1)

        table_supported = table and control_query and exact_overlap >= 1 and (

            br is not None and br <= 80 or dr is not None and dr <= 80 or sec_score > 0

        )

        accepted = strong_bm25 or dual or strong_dense or section_supported or table_supported

        # Query-specific directness estimate, for ordering only.

        directness = exact_overlap + min(expanded_overlap, 3) * 0.25

        if sec_score:

            directness += 2

        if table and control_query:

            directness += 1.5

        reasons.extend([

            reason for condition, reason in (

                (strong_bm25, "strong lexical rank and keyword overlap"),

                (dual, "dense and BM25 agreement"),

                (strong_dense, "strong dense match"),

                (section_supported, "section-aware candidate"),

                (table_supported, "table evidence passes relevance guardrail"),

            ) if condition

        ])

        candidates.append({

            "id": cid, "chunk": chunk, "score": score, "accepted": accepted,

            "directness": directness, "dense_rank": dr, "cosine_distance": dist,

            "bm25_rank": br, "bm25_score": bmscore.get(cid, 0.0),

            "keyword_coverage": round(cov, 4), "exact_term_overlap": exact_overlap,

            "section_score": round(sec_score, 5), "table_score": round(table_score, 5),

            "candidate_expansion": cid in section_expanded or cid in table_expanded,

            "guardrail_reasons": list(dict.fromkeys(reasons)) or ["failed evidence guardrails"],

        })

    candidates.sort(key=lambda x: (x["accepted"], x["score"], x["directness"]), reverse=True)

    accepted = [x for x in candidates if x["accepted"]]

    # Avoid returning several near-duplicate passages unless they are different

    # sections or table siblings. Exact duplicate content is always removed.

    chosen = []

    seen_content = set()

    for item in accepted:

        content = str(item["chunk"].get("content", "")).strip()

        if not content or content in seen_content:

            continue

        seen_content.add(content)

        chosen.append(item)

        if len(chosen) >= top_k:

            break

    results = []

    for item in chosen:

        c = item["chunk"]

        results.append({

            "chunk_id": item["id"],

            "content": c.get("content", ""),

            "source_file": c.get("source_file", ""),

            "page_start": c.get("page_start"),

            "page_end": c.get("page_end"),

            "section": c.get("section", ""),

            "chunk_kind": c.get("chunk_kind", ""),

            "element_ids": c.get("element_ids", []),

            "element_types": c.get("element_types", []),

            "table_id": c.get("table_id"),

            "retrieval": {

                "hybrid_score": round(item["score"], 6),

                "directness_score": round(item["directness"], 3),

                "dense_rank": item["dense_rank"],

                "cosine_distance": item["cosine_distance"],

                "bm25_rank": item["bm25_rank"],

                "bm25_score": round(item["bm25_score"], 4),

                "keyword_coverage": item["keyword_coverage"],

                "exact_term_overlap": item["exact_term_overlap"],

                "section_score": item["section_score"],

                "table_score": item["table_score"],

                "candidate_expansion": item["candidate_expansion"],

                "guardrail_reasons": item["guardrail_reasons"],

            },

        })

    # Evidence-based abstention: only flag no match if no candidates passed

    # the guardrails. The chat layer adds its own answer-level quote validation.

    return {

        "question": question,

        "source_filter": source,

        "candidate_count": len(candidates),

        "accepted_count": len(accepted),

        "returned_count": len(results),

        "no_reliable_match": not bool(results),

        "results": results,

    }

@asynccontextmanager

async def lifespan(app: FastAPI):

    print("Initializing improved local Harrier knowledge base...")

    if not CHUNKS_FILE.exists():

        raise RuntimeError(f"Missing chunks file: {CHUNKS_FILE}")

    if not DB_DIR.exists():

        raise RuntimeError(f"Missing ChromaDB directory: {DB_DIR}")

    raw_chunks = CHUNKS_FILE.read_bytes()

    chunks = json.loads(raw_chunks.decode("utf-8"))

    if not isinstance(chunks, list) or not chunks:

        raise RuntimeError("rag_chunks.json is empty or invalid.")

    print("Preparing BM25 index...")
    bm25_index, cache_hit = load_or_build_bm25_index(chunks, raw_chunks)
    cache_status = "loaded from persistent cache" if cache_hit else "built and cached"
    print(f"BM25 index {cache_status}: {len(chunks)} chunks.")

    client = chromadb.PersistentClient(path=str(DB_DIR))

    collection = client.get_collection(COLLECTION_NAME)

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print("Building retrieval lookup indexes...")
    retrieval_indexes = build_retrieval_indexes(chunks, bm25_index)
    by_id = {str(c["chunk_id"]): c for c in chunks}
    source_documents = sorted(retrieval_indexes["chunks_by_source"])

    STATE.update({
        "chunks": chunks,
        "by_id": by_id,
        "source_documents": source_documents,
        "bm25_index": bm25_index,
        **retrieval_indexes,
        "client": client, "collection": collection, "model": model, "ready": True,
    })

    print(
        f"Retrieval indexes ready: {len(source_documents)} source files, "
        f"{len(retrieval_indexes['section_ids_by_scope'].get('__all__', {}))} distinct sections."
    )
    print(f"Ready: {len(chunks)} chunks, {collection.count()} vectors.")

    yield

    STATE.clear()

app = FastAPI(

    title="Harrier Manual Knowledge Base",

    description="Section-aware, table-aware local hybrid retrieval API.",

    version="2.1.0",

    lifespan=lifespan,

)

class RetrievalRequest(BaseModel):

    question: str = Field(..., min_length=3, max_length=2000)

    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)

    source_file: Optional[str] = None

@app.get("/health")

def health():

    return {

        "status": "ready" if STATE.get("ready") else "starting",

        "collection": COLLECTION_NAME,

        "vector_count": STATE["collection"].count() if STATE.get("ready") else 0,

        "embedding_model": MODEL_NAME,

        "retrieval_version": "2.1.0",

    }

@app.get("/documents")

def documents():

    if not STATE.get("ready"):

        raise HTTPException(status_code=503, detail="Knowledge base is not initialized.")

    return {"documents": STATE["source_documents"]}

@app.get("/products")
def products():
    """Return the known manual/product variants available to the chat client."""
    if not STATE.get("ready"):
        raise HTTPException(status_code=503, detail="Knowledge base is not initialized.")

    labels = {
        "harrier-7-inch-infotainment-manual.json": {
            "product_id": "harrier_7inch_infotainment",
            "display_name": "Harrier 7-inch ConnectNext infotainment system",
            "aliases": ["harrier 7 inch", "7 inch infotainment", "7-inch infotainment", "connectnext", "seven inch"]
        },
        "harrier-mce-infotainment-system-user-manual-feb-2024.json": {
            "product_id": "harrier_mce_infotainment",
            "display_name": "Harrier MCE infotainment system",
            "aliases": ["harrier mce", "mce", "mce infotainment"]
        },
        "harrier-bs6-owners-manual-april-2026.json": {
            "product_id": "harrier_bs6_owner_manual",
            "display_name": "Harrier BS6 vehicle owner's manual",
            "aliases": ["harrier bs6", "bs6", "owner's manual", "owners manual", "vehicle manual"]
        },
    }
    available = []
    for source_file in STATE["source_documents"]:
        item = labels.get(source_file, {
            "product_id": source_file,
            "display_name": source_file,
            "aliases": []
        })
        available.append({**item, "source_file": source_file})
    return {"products": available}


@app.post("/retrieve")
def retrieve_endpoint(request: RetrievalRequest):

    if not STATE.get("ready"):

        raise HTTPException(status_code=503, detail="Knowledge base is not initialized.")

    try:

        return retrieve(request.question, request.top_k, request.source_file)

    except ValueError as exc:

        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:

        print(f"Retrieval error: {exc}", file=sys.stderr)

        raise HTTPException(status_code=500, detail="Retrieval failed. Check server terminal.") from exc

if __name__ == "__main__":

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
