MAX_UPLOAD_FILES = 5
MAX_UPLOAD_TOTAL_BYTES = 3 * 1024 * 1024
MAX_UPLOAD_BYTES = MAX_UPLOAD_TOTAL_BYTES

PDF_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/octet-stream",
}

MARKDOWN_CONTENT_TYPES = {
    "application/octet-stream",
    "text/markdown",
    "text/plain",
    "text/x-markdown",
}

MARKDOWN_SEPARATORS = [
    "\n# ",
    "\n## ",
    "\n### ",
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
]

REWRITE_QUERY_PROMPT = """
Rewrite the user's question into exactly three search queries for retrieving relevant
document chunks. Keep technical terms, abbreviations, and numbers. Return only a
JSON array of three strings. Do not return objects, keys, markdown, or commentary.
"""
