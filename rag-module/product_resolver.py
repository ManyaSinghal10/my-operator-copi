import json
import re
from pathlib import Path


CATALOG_PATH = Path(__file__).resolve().parent / "product_catalog.json"


def load_product_catalog():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["products"]


PRODUCTS = load_product_catalog()


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def find_product_mentions(text):
    """
    Identify product aliases explicitly mentioned in the message.

    Returns matching product records. Longer aliases are checked
    first so that specific names are preferred over generic names.
    """
    normalized_text = normalize_text(text)
    matches = {}

    aliases = []

    for product in PRODUCTS:
        for alias in product["aliases"]:
            aliases.append((alias, product))

    aliases.sort(key=lambda item: len(item[0]), reverse=True)

    for alias, product in aliases:
        normalized_alias = normalize_text(alias)

        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(normalized_alias).replace(r"\ ", r"\s+")
            + r"(?![a-z0-9])"
        )

        if re.search(pattern, normalized_text):
            matches[product["product_id"]] = product

    return list(matches.values())


def resolve_product(message, previous_product_id=None):
    """
    Resolve explicit product mentions first.

    Conversation state is used only when the current message does
    not identify a different product.
    """
    matches = find_product_mentions(message)

    if len(matches) == 1:
        return {
            "status": "EXPLICIT_PRODUCT",
            "product": matches[0]
        }

    if len(matches) > 1:
        return {
            "status": "AMBIGUOUS_PRODUCT",
            "products": matches
        }

    if previous_product_id:
        previous_product = next(
            (
                p for p in PRODUCTS
                if p["product_id"] == previous_product_id
            ),
            None
        )

        if previous_product:
            return {
                "status": "CONTEXTUAL_PRODUCT",
                "product": previous_product
            }

    return {
        "status": "NEEDS_CONTEXT",
        "products": PRODUCTS
    }