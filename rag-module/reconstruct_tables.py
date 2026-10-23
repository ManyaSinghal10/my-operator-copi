import json
import re
import hashlib

from pathlib import Path
from html.parser import HTMLParser


BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "processed" / "normalized_elements.json"
OUTPUT_FILE = BASE_DIR / "processed" / "reconstructed_tables.json"


def clean_text(text):
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


class TableParser(HTMLParser):

    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = None
        self.cell = None
        self.in_cell = False
        self.first_row_has_th = False
        self.row_has_th = False

    def handle_starttag(self, tag, attrs):

        if tag == "tr":
            self.row = []
            self.row_has_th = False

        elif tag in ("td", "th"):
            self.cell = []
            self.in_cell = True

            if tag == "th":
                self.row_has_th = True

        elif tag == "br" and self.in_cell:
            self.cell.append(" ")

    def handle_data(self, data):

        if self.in_cell and self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):

        if tag in ("td", "th"):

            self.row.append(
                clean_text(" ".join(self.cell or []))
            )

            self.cell = None
            self.in_cell = False

        elif tag == "tr":

            if self.row:
                if not self.rows:
                    self.first_row_has_th = self.row_has_th

                self.rows.append(self.row)

            self.row = None


def parse_table(html):

    parser = TableParser()

    if html:
        parser.feed(html)

    return parser.rows, parser.first_row_has_th


def normalize_row(row, column_count):

    row = list(row)

    if len(row) < column_count:
        row.extend([""] * (column_count - len(row)))

    return row


def is_empty_first_cell(row):

    return bool(row) and not clean_text(row[0])


def merge_continuation(previous_row, continuation):

    """
    Merge a continuation fragment into the previous logical row.

    Empty cells contribute nothing. Nonempty cells are appended
    to the corresponding existing cell.
    """

    column_count = max(len(previous_row), len(continuation))

    previous_row = normalize_row(previous_row, column_count)
    continuation = normalize_row(continuation, column_count)

    for index, value in enumerate(continuation):

        value = clean_text(value)

        if not value:
            continue

        if previous_row[index]:
            previous_row[index] += " " + value
        else:
            previous_row[index] = value

    return previous_row


def table_header(rows):

    if not rows:
        return []

    return [
        clean_text(cell).lower()
        for cell in rows[0]
    ]


def create_table_id(source_file, section, first_page, header):

    raw = f"{source_file}|{section}|{first_page}|{header}"

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:24]


def main():

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        elements = json.load(file)

    # Preserve source ordering.
    documents = {}

    for element in elements:

        source = element.get("source_file", "unknown")

        documents.setdefault(source, []).append(element)

    reconstructed = []

    for source_file, document_elements in documents.items():

        document_elements.sort(
            key=lambda item: item.get("element_index", 0)
        )

        fragments = []
        current_section = ""

        # -------------------------------------------------
        # Collect table fragments and their section context
        # -------------------------------------------------

        for element in document_elements:

            element_type = element.get("type")
            text = clean_text(element.get("text", ""))

            if element_type == "Title" and text:
                current_section = text

            if element_type != "Table":
                continue

            rows, has_header = parse_table(
                element.get("table_html")
            )

            if not rows:
                print(
                    "WARNING: Could not parse table:",
                    element.get("element_id")
                )
                continue

            header = rows[0]

            # The first HTML row contains column names in
            # these extracted tables.
            data_rows = rows[1:]

            fragments.append({
                "source_file": source_file,
                "section": current_section,
                "page": element.get("page_number"),
                "element_id": element.get("element_id"),
                "header": header,
                "header_key": table_header([header]),
                "has_header": has_header,
                "rows": data_rows,
            })

        # -------------------------------------------------
        # Group adjacent table fragments
        # -------------------------------------------------

        groups = []
        current_group = None

        for fragment in fragments:

            can_join = (
                current_group is not None
                and fragment["section"] == current_group["section"]
                and fragment["header_key"] == current_group["header_key"]
                and isinstance(fragment["page"], (int, float))
                and isinstance(current_group["page_end"], (int, float))
                and fragment["page"] <= current_group["page_end"] + 1
            )

            if not can_join:

                current_group = {
                    "source_file": source_file,
                    "section": fragment["section"],
                    "header": fragment["header"],
                    "header_key": fragment["header_key"],
                    "page_start": fragment["page"],
                    "page_end": fragment["page"],
                    "rows": [],
                    "element_ids": [],
                    "fragment_pages": [],
                }

                groups.append(current_group)

            current_group["page_end"] = fragment["page"]
            current_group["fragment_pages"].append(fragment["page"])
            current_group["element_ids"].append(
                fragment["element_id"]
            )

            fragment_rows = fragment["rows"]

            for row_index, row in enumerate(fragment_rows):

                row = normalize_row(
                    row,
                    len(current_group["header"])
                )

                # Only consider the first data row of a
                # subsequent page for continuation merging.
                is_first_row_of_continuation = (
                    can_join and row_index == 0
                )

                if (
                    is_first_row_of_continuation
                    and is_empty_first_cell(row)
                    and current_group["rows"]
                ):
                    current_group["rows"][-1] = merge_continuation(
                        current_group["rows"][-1],
                        row,
                    )

                else:
                    current_group["rows"].append(row)

        # -------------------------------------------------
        # Produce reconstructed table records
        # -------------------------------------------------

        for group in groups:

            group["table_id"] = create_table_id(
                group["source_file"],
                group["section"],
                group["page_start"],
                group["header"],
            )

            group.pop("header_key", None)

            reconstructed.append(group)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            reconstructed,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 60)
    print("TABLE RECONSTRUCTION COMPLETE")
    print("=" * 60)
    print(f"Logical tables: {len(reconstructed)}")
    print(f"Output: {OUTPUT_FILE}")

    print("\nTables spanning multiple pages:")

    for table in reconstructed:

        pages = table.get("fragment_pages", [])

        if len(set(pages)) > 1:
            print(
                f"{table['source_file']} | "
                f"{table['section']} | "
                f"pages {table['page_start']}-{table['page_end']} | "
                f"rows {len(table['rows'])}"
            )


if __name__ == "__main__":
    main()