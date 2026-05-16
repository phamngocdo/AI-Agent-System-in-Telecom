import json
import re
from typing import Optional, Sequence


def dedupe(items: Sequence[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        normalized = str(item).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def parse_json_string_list(text: str) -> list[str]:
    parsed = extract_json(text)
    if isinstance(parsed, list):
        return [
            normalized
            for item in parsed
            if (normalized := normalize_query_item(item))
        ]
    return []


def normalize_query_item(item) -> str:
    if isinstance(item, str):
        return item.strip()

    if isinstance(item, dict):
        for key in ("query", "text", "search_query", "q"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return ""


def extract_json(text: str):
    normalized = (text or "").strip()
    if not normalized:
        return None
    try:
        return json.loads(normalized)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", normalized)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def tokenize_terms(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[\wÀ-ỹ]+", (text or "").lower())
        if len(token) > 1
    ]


def tokenize(text: str) -> set[str]:
    return set(tokenize_terms(text))


def format_history(
    conversation_history: Optional[Sequence[tuple[str, str]]],
    max_turns: int = 4,
    max_chars: int = 700,
) -> str:
    if not conversation_history:
        return ""

    lines = []
    for role, content in conversation_history[-max_turns:]:
        lines.append(f"{role}: {str(content).strip()[:max_chars]}")
    return "\n".join(lines)


def safe_pdf_filename(filename: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename or "document.pdf")
    if not stem.lower().endswith(".pdf"):
        stem = f"{stem}.pdf"
    return stem
