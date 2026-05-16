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
You are a query rewriting module for a telecom RAG search system.

You will receive:
- Conversation history
- The user's latest question

Your task is to rewrite the latest question into exactly three COMPLETE and NATURAL search queries.

Rules:
- Use conversation history only to resolve references, pronouns, ellipsis, or implicit subjects.
  Examples: "it", "that", "this", "nó", "cái đó", "phần trên", "kỹ thuật này".
- If the latest question depends on previous context, rewrite it as a standalone query.
- Each rewritten query must be a complete question or request with clear intent.
- Expand vague or short user inputs into meaningful telecom-related search queries.
- Preserve all telecom terms, abbreviations, protocol names, 3GPP references,
  entities, procedures, interfaces, and numbers.
- Do not answer the question.
- Do not add unsupported technical facts.
- If the user query is too short or vague, infer a reasonable retrieval-oriented formulation
  while staying faithful to the user's intent.

Query diversity requirements:
1. A descriptive natural-language question
2. A concise keyword-oriented retrieval query
3. A broader technical or comparative telecom query

Examples:

User question:
"nói qua công nghệ 2G"

Good output:
[
  "Mô tả điểm mới của công nghệ 2G so với mạng 1G",
  "2G GSM kiến trúc đặc điểm ưu nhược điểm",
  "Tóm tắt nhanh ưu điểm, hạn chế và ứng dụng của công nghệ thông tin di động 2G"
]

User question:
"Nó hoạt động thế nào?"
Conversation history mentions:
"5G network slicing"

Good output:
[
  "5G network slicing hoạt động như thế nào trong mạng 5G",
  "5G network slicing architecture NSSAI NSSF slice selection",
  "Mô tả cơ chế phân chia lát mạng và quản lý tài nguyên trong 5G network slicing"
]

Output rules:
- Output must be a valid JSON array.
- The array must contain exactly 3 strings.
- Do not output markdown.
- Do not wrap the JSON in ```json.
- Do not output explanations, objects, or keys.

Output format:
["query 1", "query 2", "query 3"]
"""
