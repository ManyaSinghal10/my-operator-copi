import json
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "processed" / "normalized_elements.json"


def main():
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        elements = json.load(file)

    print("=" * 60)
    print("NORMALIZED DATA QUALITY REPORT")
    print("=" * 60)

    # -----------------------------------------------------
    # 1. Count elements by document and type
    # -----------------------------------------------------

    document_counts = defaultdict(Counter)

    for element in elements:
        document = element.get("source_file", "Unknown")
        element_type = element.get("type", "Unknown")

        document_counts[document][element_type] += 1

    print("\n1. ELEMENT COUNTS BY DOCUMENT")

    for document, counts in document_counts.items():
        print(f"\nDocument: {document}")

        for element_type, count in counts.most_common():
            print(f"  {element_type}: {count}")

    # -----------------------------------------------------
    # 2. Inspect empty elements
    # -----------------------------------------------------

    empty_elements = [
        element
        for element in elements
        if not element.get("text", "").strip()
    ]

    empty_counts = Counter(
        element.get("type", "Unknown")
        for element in empty_elements
    )

    print("\n2. EMPTY TEXT ELEMENTS")
    print(f"Total: {len(empty_elements)}")
    print("Distribution:", dict(empty_counts))

    print("\nFirst 20 empty elements:")

    for element in empty_elements[:20]:
        print({
            "source_file": element.get("source_file"),
            "type": element.get("type"),
            "page_number": element.get("page_number"),
            "element_id": element.get("element_id"),
            "element_index": element.get("element_index"),
        })

    # -----------------------------------------------------
    # 3. Inspect tables
    # -----------------------------------------------------

    tables = [
        element
        for element in elements
        if element.get("type") == "Table"
    ]

    tables_without_html = [
        element
        for element in tables
        if not element.get("table_html")
    ]

    tables_without_text = [
        element
        for element in tables
        if not element.get("text", "").strip()
    ]

    print("\n3. TABLE QUALITY")
    print(f"Total tables: {len(tables)}")
    print(f"Tables without HTML: {len(tables_without_html)}")
    print(f"Tables without text: {len(tables_without_text)}")

    print("\nFirst 5 tables:")

    for table in tables[:5]:
        print({
            "source_file": table.get("source_file"),
            "page_number": table.get("page_number"),
            "element_id": table.get("element_id"),
            "text_preview": table.get("text", "")[:250],
            "has_html": bool(table.get("table_html")),
        })

    # -----------------------------------------------------
    # 4. Inspect image elements
    # -----------------------------------------------------

    images = [
        element
        for element in elements
        if element.get("type") == "Image"
    ]

    images_with_text = [
        element
        for element in images
        if element.get("text", "").strip()
    ]

    print("\n4. IMAGE QUALITY")
    print(f"Total image elements: {len(images)}")
    print(f"Images with text: {len(images_with_text)}")
    print(f"Images without text: {len(images) - len(images_with_text)}")

    print("\nFirst 15 image descriptions:")

    for image in images[:15]:
        print({
            "source_file": image.get("source_file"),
            "page_number": image.get("page_number"),
            "text": image.get("text", "")[:150],
        })

    # -----------------------------------------------------
    # 5. Check page-number availability
    # -----------------------------------------------------

    missing_pages = [
        element
        for element in elements
        if element.get("page_number") is None
    ]

    print("\n5. PAGE NUMBER QUALITY")
    print(f"Elements without page numbers: {len(missing_pages)}")

    # -----------------------------------------------------
    # 6. Check duplicate element IDs
    # -----------------------------------------------------

    id_counts = Counter(
        (
            element.get("source_file"),
            element.get("element_id")
        )
        for element in elements
    )

    duplicates = {
        key: count
        for key, count in id_counts.items()
        if key[1] and count > 1
    }

    print("\n6. ELEMENT ID QUALITY")
    print(f"Duplicate IDs within a document: {len(duplicates)}")

    for key, count in list(duplicates.items())[:10]:
        print(f"  {key}: {count} occurrences")

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()