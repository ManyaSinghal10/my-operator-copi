import json
import re
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"

ELEMENTS_FILE = PROCESSED_DIR / "normalized_elements.json"
TABLES_FILE = PROCESSED_DIR / "reconstructed_tables.json"
OUTPUT_FILE = PROCESSED_DIR / "rag_chunks.json"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Character-based starting values, not token limits.
TARGET_CHUNK_CHARS = 900
MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 300

# Cosine distance = 1 - cosine similarity.
SEMANTIC_DISTANCE_THRESHOLD = 0.32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# GENERAL UTILITIES
# =========================================================

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def make_id(value):
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()[:24]


def valid_page(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    )


def split_sentences(text):
    """
    Basic English-oriented sentence segmentation.
    Review abbreviations and unusual punctuation during validation.
    """

    text = clean_text(text)

    if not text:
        return []

    sentences = re.split(
        r'(?<=[.!?])\s+(?=[A-Z0-9"\[(])',
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def meaningful_image_description(text):
    text = clean_text(text)

    if not text:
        return False

    if re.fullmatch(r"\d+", text):
        return False

    ignored = {
        "tata logo",
        "harrier logo",
        "connectnext logo",
    }

    return text.lower() not in ignored


# =========================================================
# ELEMENT UNITS
# =========================================================

def make_unit(text, element):
    return {
        "text": clean_text(text),
        "page": element.get("page_number"),
        "element_id": element.get("element_id"),
        "type": element.get("type", "Unknown"),
    }


def element_to_units(element):
    """
    Split narrative elements into sentences for semantic analysis.
    Keep list items, captions, and image descriptions intact.
    """

    element_type = element.get("type", "Unknown")
    text = clean_text(element.get("text", ""))

    if not text:
        return []

    if element_type == "NarrativeText":
        sentences = split_sentences(text)

        if sentences:
            return [
                make_unit(sentence, element)
                for sentence in sentences
            ]

    if element_type == "Image":

        if not meaningful_image_description(text):
            return []

        text = f"[Image description: {text}]"

    elif element_type == "FigureCaption":
        text = f"[Figure caption: {text}]"

    return [make_unit(text, element)]


# =========================================================
# SEMANTIC CHUNKING FOR NARRATIVE CONTENT
# =========================================================

def semantic_chunk_units(
    units,
    source_file,
    section,
    model,
    chunk_number,
):
    """
    Semantic splitting applies to narrative and contextual
    text units, not reconstructed tables.

    Section boundaries are handled by the caller.
    """

    if not units:
        return [], chunk_number

    texts = [unit["text"] for unit in units]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    chunks = []
    current_units = []
    current_length = 0

    def flush():

        nonlocal current_units
        nonlocal current_length
        nonlocal chunk_number

        if not current_units:
            return

        body = "\n".join(
            unit["text"]
            for unit in current_units
        ).strip()

        content = (
            f"Section: {section}\n\n{body}"
            if section
            else body
        )

        pages = [
            unit["page"]
            for unit in current_units
            if valid_page(unit["page"])
        ]

        element_ids = list(dict.fromkeys(
            unit["element_id"]
            for unit in current_units
            if unit["element_id"]
        ))

        element_types = list(dict.fromkeys(
            unit["type"]
            for unit in current_units
        ))

        identity = (
            f"{source_file}|{chunk_number}|"
            f"{min(pages) if pages else None}|"
            f"{max(pages) if pages else None}|{content}"
        )

        chunks.append({
            "chunk_id": make_id(identity),
            "source_file": source_file,
            "page_start": min(pages) if pages else None,
            "page_end": max(pages) if pages else None,
            "section": section,
            "content": content,
            "element_ids": element_ids,
            "element_types": element_types,
            "chunk_kind": "narrative",
        })

        chunk_number += 1
        current_units = []
        current_length = 0

    previous_embedding = None

    for index, unit in enumerate(units):

        text = unit["text"]
        length = len(text)

        semantic_distance = 0.0

        if previous_embedding is not None:
            similarity = float(
                np.dot(previous_embedding, embeddings[index])
            )

            similarity = max(-1.0, min(1.0, similarity))
            semantic_distance = 1.0 - similarity

        exceeds_max = (
            current_units
            and current_length + length + 1 > MAX_CHUNK_CHARS
        )

        semantic_boundary = (
            current_units
            and current_length >= MIN_CHUNK_CHARS
            and semantic_distance >= SEMANTIC_DISTANCE_THRESHOLD
        )

        target_boundary = (
            current_units
            and current_length >= TARGET_CHUNK_CHARS
        )

        if exceeds_max or semantic_boundary or target_boundary:
            flush()

        # Preserve oversized units without truncating them.
        # Long sentences are divided into character-sized pieces.
        if length > MAX_CHUNK_CHARS:

            flush()

            parts = split_sentences(text)

            if len(parts) <= 1:
                parts = [
                    text[i:i + MAX_CHUNK_CHARS]
                    for i in range(0, len(text), MAX_CHUNK_CHARS)
                ]

            for part in parts:

                if len(part) > MAX_CHUNK_CHARS:
                    subparts = [
                        part[i:i + MAX_CHUNK_CHARS]
                        for i in range(
                            0, len(part), MAX_CHUNK_CHARS
                        )
                    ]
                else:
                    subparts = [part]

                for subpart in subparts:
                    current_units.append({
                        **unit,
                        "text": subpart,
                    })
                    current_length += len(subpart) + 1
                    flush()

            previous_embedding = embeddings[index]
            continue

        current_units.append(unit)
        current_length += length + 1

        previous_embedding = embeddings[index]

    flush()

    return chunks, chunk_number


# =========================================================
# RECONSTRUCTED TABLE CHUNKING
# =========================================================

def format_row(row):
    return " | ".join(
        clean_text(str(cell))
        for cell in row
    )


def table_to_chunks(
    table,
    source_file,
    chunk_number,
):
    """
    Consume a reconstructed logical table.

    Split only between complete rows. A single row can exceed
    the configured target size if necessary to preserve meaning.
    """

    rows = table.get("rows", [])

    if not rows:
        return [], chunk_number

    section = table.get("section", "")
    header = table.get("header", [])

    # A table header provides useful context but is not
    # automatically assumed to be a data row.
    header_text = format_row(header) if header else ""

    caption = table.get("caption", "")
    page_start = table.get("page_start")
    page_end = table.get("page_end")

    row_groups = []
    current_rows = []
    current_length = len(header_text)

    for row in rows:

        row_text = format_row(row)

        projected_length = (
            current_length + len(row_text) + 1
        )

        if (
            current_rows
            and projected_length > MAX_CHUNK_CHARS
        ):
            row_groups.append(current_rows)
            current_rows = []
            current_length = len(header_text)

        # Keep every logical row intact, even if that row alone
        # exceeds MAX_CHUNK_CHARS.
        current_rows.append(row)
        current_length += len(row_text) + 1

    if current_rows:
        row_groups.append(current_rows)

    chunks = []

    for group_index, group in enumerate(row_groups):

        content_parts = []

        if section:
            content_parts.append(f"Section: {section}")

        if caption:
            content_parts.append(f"Table caption: {caption}")

        if header_text:
            content_parts.append(f"Columns: {header_text}")

        content_parts.append(
            "\n".join(format_row(row) for row in group)
        )

        content = "\n".join(content_parts)

        identity = (
            f"{source_file}|{chunk_number}|"
            f"{table.get('table_id', '')}|"
            f"{group_index}|{content}"
        )

        chunks.append({
            "chunk_id": make_id(identity),
            "source_file": source_file,
            "page_start": page_start,
            "page_end": page_end,
            "section": section,
            "content": content,
            "element_ids": table.get("element_ids", []),
            "element_types": ["Table"],
            "chunk_kind": "table",
            "table_id": table.get("table_id"),
            "table_part": group_index + 1,
            "table_parts": len(row_groups),
        })

        chunk_number += 1

    return chunks, chunk_number


# =========================================================
# DOCUMENT EVENT PREPARATION
# =========================================================

def prepare_document_events(elements, tables):

    element_by_id = {
        element.get("element_id"): element
        for element in elements
        if element.get("element_id")
    }

    table_fragment_ids = set()
    table_events = []

    for table in tables:

        fragment_ids = table.get("element_ids", [])
        table_fragment_ids.update(fragment_ids)

        indexes = [
            element_by_id[element_id].get("element_index")
            for element_id in fragment_ids
            if element_id in element_by_id
            and isinstance(
                element_by_id[element_id].get("element_index"),
                int,
            )
        ]

        insert_index = min(indexes) if indexes else 10**12

        table_events.append({
            "event_type": "Table",
            "element_index": insert_index,
            "table": table,
        })

    # Find the closest preceding table for each table caption.
    captions = [
        element for element in elements
        if element.get("type") == "FigureCaption"
        and re.match(
            r"^\s*Table\b",
            clean_text(element.get("text", "")),
            re.I,
        )
    ]

    attached_caption_ids = set()

    for caption in captions:

        caption_page = caption.get("page_number")
        caption_index = caption.get("element_index", 0)
        source_file = caption.get("source_file")

        candidates = []

        for table in tables:

            if table.get("source_file") != source_file:
                continue

            if table.get("page_end") != caption_page:
                continue

            fragment_indexes = [
                element_by_id[element_id].get("element_index")
                for element_id in table.get("element_ids", [])
                if element_id in element_by_id
                and isinstance(
                    element_by_id[element_id].get("element_index"),
                    int,
                )
            ]

            if not fragment_indexes:
                continue

            last_fragment_index = max(fragment_indexes)

            if last_fragment_index <= caption_index:
                candidates.append(
                    (last_fragment_index, table)
                )

        if candidates:

            _, matched_table = max(
                candidates,
                key=lambda item: item[0],
            )

            matched_table["caption"] = clean_text(
                caption.get("text", "")
            )

            attached_caption_ids.add(
                caption.get("element_id")
            )

    events = []

    for element in elements:

        element_id = element.get("element_id")

        if element_id in table_fragment_ids:
            continue

        if element_id in attached_caption_ids:
            continue

        events.append({
            "event_type": "Element",
            "element_index": element.get("element_index", 0),
            "element": element,
        })

    events.extend(table_events)

    events.sort(
        key=lambda event: event.get("element_index", 0)
    )

    return events


# =========================================================
# PROCESS ONE DOCUMENT
# =========================================================

def process_document(elements, tables, source_file, model):

    events = prepare_document_events(elements, tables)

    chunks = []
    section = ""
    pending_units = []
    chunk_number = 0

    def flush_pending():

        nonlocal pending_units
        nonlocal chunk_number

        if not pending_units:
            return

        new_chunks, chunk_number = semantic_chunk_units(
            units=pending_units,
            source_file=source_file,
            section=section,
            model=model,
            chunk_number=chunk_number,
        )

        chunks.extend(new_chunks)
        pending_units = []

    for event in events:

        if event["event_type"] == "Table":

            flush_pending()

            table_chunks, chunk_number = table_to_chunks(
                table=event["table"],
                source_file=source_file,
                chunk_number=chunk_number,
            )

            chunks.extend(table_chunks)
            continue

        element = event["element"]
        element_type = element.get("type", "Unknown")
        text = clean_text(element.get("text", ""))

        if element_type == "Title":

            flush_pending()

            if text:
                section = text

            continue

        # These are excluded from standalone retrieval chunks.
        if element_type in ("PageNumber", "Header", "Form"):
            continue

        if element_type in (
            "NarrativeText",
            "ListItem",
            "Image",
            "FigureCaption",
        ):
            pending_units.extend(
                element_to_units(element)
            )

    flush_pending()

    return chunks


# =========================================================
# MAIN
# =========================================================

def main():

    for path in (ELEMENTS_FILE, TABLES_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}"
            )

    with ELEMENTS_FILE.open("r", encoding="utf-8") as file:
        elements = json.load(file)

    with TABLES_FILE.open("r", encoding="utf-8") as file:
        reconstructed_tables = json.load(file)

    elements_by_document = defaultdict(list)
    tables_by_document = defaultdict(list)

    for element in elements:
        elements_by_document[
            element.get("source_file", "unknown")
        ].append(element)

    for table in reconstructed_tables:
        tables_by_document[
            table.get("source_file", "unknown")
        ].append(table)

    # Load the embedding model once.
    print(f"Loading semantic model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")

    model = SentenceTransformer(
        MODEL_NAME,
        device=DEVICE,
    )

    all_chunks = []

    for source_file, document_elements in elements_by_document.items():

        print(f"\nProcessing: {source_file}")

        document_chunks = process_document(
            elements=document_elements,
            tables=tables_by_document.get(source_file, []),
            source_file=source_file,
            model=model,
        )

        all_chunks.extend(document_chunks)

        print(f"Input elements: {len(document_elements)}")
        print(f"Reconstructed tables: "
              f"{len(tables_by_document.get(source_file, []))}")
        print(f"Generated chunks: {len(document_chunks)}")

    # Validate IDs.
    chunk_ids = [
        chunk["chunk_id"]
        for chunk in all_chunks
    ]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise RuntimeError("Duplicate chunk IDs detected.")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            all_chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 60)
    print("HYBRID CHUNKING COMPLETE")
    print("=" * 60)
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Unique chunk IDs: {len(set(chunk_ids))}")
    print(f"Output: {OUTPUT_FILE}")

    print("\nChunk-type counts:")

    for kind in ("narrative", "table"):
        count = sum(
            1 for chunk in all_chunks
            if chunk.get("chunk_kind") == kind
        )
        print(f"{kind}: {count}")

    print("\nFirst 3 chunks:")

    for chunk in all_chunks[:3]:
        print(json.dumps(
            chunk,
            ensure_ascii=False,
            indent=2,
        ))


if __name__ == "__main__":
    main()