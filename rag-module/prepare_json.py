import json
import re
from pathlib import Path
from collections import Counter
from html.parser import HTMLParser


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# Change this if your folder has a different name.
JSON_DIR = BASE_DIR / "JSON_files"

OUTPUT_DIR = BASE_DIR / "processed"

OUTPUT_FILE = OUTPUT_DIR / "normalized_elements.json"


# =========================================================
# HTML TABLE CONVERTER
# Converts table HTML into readable text while preserving
# row and cell boundaries.
# =========================================================

class TableHTMLParser(HTMLParser):

    def __init__(self):
        super().__init__()
        self.parts = []
        self.cell_has_content = False

    def handle_starttag(self, tag, attrs):

        if tag in ("td", "th"):
            if self.cell_has_content:
                self.parts.append(" | ")

            self.cell_has_content = False

        elif tag == "tr":
            if self.parts and not self.parts[-1].endswith("\n"):
                self.parts.append("\n")

        elif tag in ("br", "p"):
            self.parts.append("\n")

    def handle_endtag(self, tag):

        if tag in ("td", "th"):
            self.cell_has_content = True

        elif tag == "tr":
            self.parts.append("\n")

        elif tag == "p":
            self.parts.append("\n")

    def handle_data(self, data):
        text = data.strip()

        if text:
            self.parts.append(text)
            self.cell_has_content = True

    def get_text(self):
        text = "".join(self.parts)

        lines = []

        for line in text.splitlines():
            line = re.sub(r"\s+", " ", line).strip()

            if line:
                lines.append(line)

        return "\n".join(lines)


def html_to_text(html):
    """Convert HTML into readable text."""

    if not html:
        return ""

    parser = TableHTMLParser()
    parser.feed(html)

    return parser.get_text()


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text):
    """
    Clean unnecessary whitespace without removing
    meaningful text or changing its wording.
    """

    if not isinstance(text, str):
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\x00", "")

    # Normalize whitespace within each line.
    lines = []

    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)

    text = "\n".join(lines)

    # Reduce excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# =========================================================
# PROCESS ONE JSON FILE
# =========================================================

def process_json_file(json_path):

    print(f"\nProcessing: {json_path.name}")

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    elements = data.get("elements", [])

    if not isinstance(elements, list):
        print("WARNING: 'elements' is not a list.")
        return []

    document_id = data.get("id", json_path.stem)

    normalized_elements = []

    for index, element in enumerate(elements):

        if not isinstance(element, dict):
            continue

        element_type = element.get("type", "Unknown")

        metadata = element.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {}

        text = normalize_text(element.get("text", ""))

        table_html = metadata.get("text_as_html", "")

        # Prefer HTML-derived text for tables when available.
        if element_type == "Table" and table_html:
            table_text = html_to_text(table_html)

            if table_text:
                text = table_text

        # Keep records even when their text is empty.
        # They may be useful for diagnostics.
        record = {
            "document_id": document_id,
            "source_file": json_path.name,
            "element_id": element.get(
                "element_id",
                f"{json_path.stem}_element_{index}"
            ),
            "element_index": index,
            "type": element_type,
            "page_number": metadata.get("page_number"),
            "text": text,
            "table_html": (
                table_html if element_type == "Table" else None
            ),
        }

        normalized_elements.append(record)

    print(f"Elements extracted: {len(normalized_elements)}")

    return normalized_elements


# =========================================================
# MAIN
# =========================================================

def main():

    if not JSON_DIR.exists():
        raise FileNotFoundError(
            f"JSON directory not found: {JSON_DIR}\n"
            "Update JSON_DIR in prepare_json.py."
        )

    json_files = sorted(JSON_DIR.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in: {JSON_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_elements = []

    for json_path in json_files:

        try:
            elements = process_json_file(json_path)
            all_elements.extend(elements)

        except (json.JSONDecodeError, OSError) as error:
            print(f"ERROR processing {json_path.name}: {error}")

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            all_elements,
            file,
            ensure_ascii=False,
            indent=2
        )

    # Summary
    type_counts = Counter(
        element["type"] for element in all_elements
    )

    print("\n" + "=" * 50)
    print("JSON NORMALIZATION COMPLETE")
    print("=" * 50)

    print(f"Input JSON files: {len(json_files)}")
    print(f"Total elements: {len(all_elements)}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nElement type distribution:")

    for element_type, count in type_counts.most_common():
        print(f"{element_type}: {count}")

    empty_text_count = sum(
        1 for element in all_elements
        if not element["text"]
    )

    print(f"\nElements with empty text: {empty_text_count}")


if __name__ == "__main__":
    main()