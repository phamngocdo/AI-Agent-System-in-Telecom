import asyncio
import re
import xml.etree.ElementTree as ET

import httpx

BATCH_SIZE = 100
MAX_CONCURRENT = 10
API_URL = "https://export.arxiv.org/api/query?id_list={ids}&max_results={n}"
NS = {"atom": "http://www.w3.org/2005/Atom"}

def normalize_id(arxiv_id: str) -> str:
    """Strip the version suffix from an arXiv ID.

    Args:
        arxiv_id: Raw arXiv identifier, optionally with a version tag.

    Returns:
        Version-free arXiv identifier.
    """
    return re.sub(r"v\d+$", "", arxiv_id)


def extract_arxiv_id(record: dict) -> str | None:
    """Extract a normalised arXiv ID from a JSONL record.

    Lookup order:
        1. ``meta.arxiv_id`` field (preferred).
        2. URL inside ``meta.url``, parsed for new-style (``YYMM.NNNNN``)
           or old-style (``cat/NNNNNNN``) IDs.

    Args:
        record: A single decoded JSONL object.

    Returns:
        Normalised arXiv ID string, or ``None`` if not found.
    """
    arxiv_id = record.get("meta", {}).get("arxiv_id")
    if arxiv_id:
        return normalize_id(str(arxiv_id))

    url = record.get("meta", {}).get("url", "")
    if not url:
        return None

    match = re.search(r"arxiv\.org/(?:abs|pdf)/([^?#\s]+)", url)
    if not match:
        return None

    raw = match.group(1).rstrip("/")

    # New-style: 2301.12345 or 2301.12345v2
    if re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", raw):
        return normalize_id(raw)

    # Old-style: cs/0601009 or cs.AI/0601009v1
    if re.fullmatch(r"[a-z][a-z0-9\-]*(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?", raw):
        return normalize_id(raw)

    return None


async def _fetch_batch(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    batch_ids: list[str],
    batch_idx: int,
    total_batches: int,
) -> dict[str, str]:
    """Fetch paper titles for one batch of arXiv IDs.

    Args:
        client: Shared async HTTP client.
        sem: Semaphore controlling concurrency.
        batch_ids: List of normalised arXiv IDs (≤ ``BATCH_SIZE``).
        batch_idx: Zero-based index of this batch (used for logging).
        total_batches: Total number of batches (used for logging).

    Returns:
        Mapping of ``arxiv_id → title`` for every paper found in the
        API response.
    """
    url = _API_URL.format(ids=",".join(batch_ids), n=len(batch_ids))

    async with sem:
        response = await client.get(url)
        response.raise_for_status()

    root = ET.fromstring(response.text)
    result: dict[str, str] = {}

    for entry in root.findall("atom:entry", _NS):
        id_elem = entry.find("atom:id", _NS)
        title_elem = entry.find("atom:title", _NS)
        if id_elem is None or title_elem is None:
            continue

        paper_id = normalize_id(id_elem.text.split("/abs/")[-1])
        title = title_elem.text.strip().replace("\n", " ")
        result[paper_id] = title

    print(f"  [{batch_idx + 1}/{total_batches}] fetched {len(result)} titles")
    return result


async def fetch_titles(arxiv_ids: list[str]) -> dict[str, str]:
    """Fetch titles for a collection of arXiv IDs concurrently.

    Args:
        arxiv_ids: Unique, normalised arXiv IDs to look up.

    Returns:
        Mapping of ``arxiv_id → title`` for every successfully resolved ID.
    """
    if not arxiv_ids:
        return {}

    batches = [
        arxiv_ids[i: i + BATCH_SIZE]
        for i in range(0, len(arxiv_ids), BATCH_SIZE)
    ]
    total_batches = len(batches)
    print(f"Fetching titles: {len(arxiv_ids)} IDs in {total_batches} batches "
          f"(concurrency={MAX_CONCURRENT})")

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    title_map: dict[str, str] = {}

    async with httpx.AsyncClient(
        timeout=30,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0"},
    ) as client:
        tasks = [
            _fetch_batch(client, sem, batch, idx, total_batches)
            for idx, batch in enumerate(batches)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, dict):
            title_map.update(result)
        else:
            print(f"  [WARN] batch failed: {result}")

    print(f"Titles resolved: {len(title_map)} / {len(arxiv_ids)}")
    return title_map


def fetch_titles_sync(arxiv_ids: list[str]) -> dict[str, str]:
    """Synchronous wrapper around :func:`fetch_titles`.
    Args:
        arxiv_ids: Unique, normalised arXiv IDs to look up.

    Returns:
        Mapping of ``arxiv_id → title``.
    """
    return asyncio.run(fetch_titles(arxiv_ids))
