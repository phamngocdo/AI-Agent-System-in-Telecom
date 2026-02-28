import json
import re

from src.data.cleaner.base_cleaner import BaseCleaner
from src.utils.arxiv_utils import extract_arxiv_id, fetch_titles_sync


class ArxivLatexCleaner(BaseCleaner):
    """Clean LaTeX-based arXiv JSONL files and split them into sections.

    Pipeline:
        1. Collect unique arXiv IDs from all input records.
        2. Fetch paper titles via the arXiv Atom API (concurrent batches).
        3. For each record, split the raw LaTeX into sections.
        4. Clean each section (remove commands, environments, noise).
        5. Drop sections shorter than ``MIN_WORDS`` words.
        6. Write ``{"text": ..., "file_name": <title>}`` records to output.

    Attributes:
        SECTION_PATTERN (str): Regex matching LaTeX section/subsection headings.
        MIN_WORDS (int): Minimum word count required to keep a section.
    """

    SECTION_PATTERN = r"\\(?:sub)*section\*?\{(.*?)\}"
    MIN_WORDS = 40

    def split_sections_raw(self, text: str) -> list[tuple[str, str]]:
        """Split raw LaTeX text into (title, content) pairs by section headings.

        Args:
            text: Raw LaTeX source string.

        Returns:
            List of ``(section_title, section_content)`` tuples in document order.
            Returns an empty list when no headings are found.
        """
        matches = list(re.finditer(self.SECTION_PATTERN, text))
        if not matches:
            return []

        sections: list[tuple[str, str]] = []
        for i, match in enumerate(matches):
            title = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            sections.append((title, content))

        return sections

    def clean_text(self, text: str) -> str:
        """Remove LaTeX markup and normalise whitespace.

        Removes:
            - References section and everything after it.
            - ``\\begin`` / ``\\end`` environments (multiline).
            - Citations (``\\cite``, ``\\ref``, ``\\eqref``, ``\\label``).
            - Inline math (``$...$``).
            - URLs.
            - LaTeX commands, preserving their argument text where possible.
            - LaTeX comments (``%...``).

        Args:
            text: Raw LaTeX string.

        Returns:
            Cleaned plain-text string.
        """
        text = text.replace("\r\n", "\n")

        # Drop references section and everything that follows
        text = re.sub(
            r"\\section\*?\{[Rr]eferences\}.*",
            "",
            text,
            flags=re.DOTALL,
        )

        # Remove block environments
        text = re.sub(
            r"\\begin\{.*?\}.*?\\end\{.*?\}",
            "",
            text,
            flags=re.DOTALL,
        )

        # Remove citation and cross-reference commands
        text = re.sub(r"\\cite[tp]?\{.*?\}", "", text)
        text = re.sub(r"\\ref\{.*?\}", "", text)
        text = re.sub(r"\\eqref\{.*?\}", "", text)
        text = re.sub(r"\\label\{.*?\}", "", text)

        # Expand IEEE paragraph-start macro
        text = re.sub(r"\\IEEEPARstart\{(.?)\}\{(.*?)\}", r"\1\2", text)

        # Remove inline math
        text = re.sub(r"\$[^$]*\$", "", text)

        # Remove URLs
        text = re.sub(r"http\S+", "", text)

        # Expand commands that wrap text (e.g. \textbf{foo} → foo)
        text = re.sub(r"\\[a-zA-Z]+\*?\{(.*?)\}", r"\1", text)

        # Drop remaining bare commands (e.g. \noindent)
        text = re.sub(r"\\[a-zA-Z]+\*?", "", text)

        # Remove LaTeX line comments
        text = re.sub(r"%.*", "", text)

        # Normalise whitespace
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()

    def run(self) -> None:
        """Run the full cleaning pipeline and write results to the output file.

        Steps:
            1. Scan all ``*.jsonl`` files under ``input_path``.
            2. Fetch paper titles from the arXiv API for all collected IDs.
            3. Clean and split each document; write surviving sections.
        """
        files = list(self.input_path.rglob("*.jsonl"))
        print(f"Found {len(files)} input file(s)")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        records: list[dict] = []
        unique_ids: set[str] = set()

        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    records.append(obj)
                    arxiv_id = extract_arxiv_id(obj)
                    if arxiv_id:
                        unique_ids.add(arxiv_id)

        print(f"Loaded {len(records)} records, {len(unique_ids)} unique arXiv IDs")

        title_map = fetch_titles_sync(list(unique_ids))

        total_output = 0
        total_removed = 0

        with open(self.output_path, "w", encoding="utf-8") as f_out:
            for obj in records:
                raw_text = obj.get("text", "")
                if not raw_text:
                    continue

                arxiv_id = extract_arxiv_id(obj)
                file_name = title_map.get(arxiv_id, "") if arxiv_id else ""

                for _section_title, content in self.split_sections_raw(raw_text):
                    cleaned = self.clean_text(content)

                    if len(cleaned.split()) < self.MIN_WORDS:
                        total_removed += 1
                        continue

                    f_out.write(
                        json.dumps(
                            {"text": cleaned, "file_name": file_name},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    total_output += 1

        print(
            f"Done. Sections written: {total_output} "
            f"(skipped {total_removed} short sections)"
        )


if __name__ == "__main__":
    cleaner = ArxivLatexCleaner(
        input_dir="data/pretrain_data/raw/redpajama/arxiv",
        output_file="data/pretrain_data/processed/redpajama/arxiv.jsonl",
    )
    cleaner.run()
