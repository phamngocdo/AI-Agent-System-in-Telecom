import re
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.data.cleaner.base_cleaner import BaseCleaner


class ThreeGPPCleaner(BaseCleaner):
    """Clean 3GPP Markdown specification files into section-level JSONL records.

    Pipeline per document:
        1. Trims content to the range [Scope section ... Annex].
        2. Splits content into hierarchical section blocks.
        3. Removes unwanted top-level sections (Scope, References).
        4. Normalizes paragraphs, removes figures/tables, cleans spacing.
        5. Filters out references, Markdown syntax, and short blocks.

    Output is organized per release into separate JSONL files inside the
    output directory.
    """

    def _trim_by_scope_and_annex(self, lines: list[str]) -> list[str] | None:
        """Trim document lines to keep only content between Scope and Annex.

        Locates the "1 Scope" heading 
        and stops before the first Annex heading.

        Args:
            lines: Raw lines read from a Markdown file.

        Returns:
            Trimmed list of lines, or None if no Scope heading is found.
        """
        scope_index = None

        for i, line in enumerate(lines):
            if re.match(r"^\s*#+\s*1\s+Scope\s*\n?$", line, re.IGNORECASE):
                scope_index = i
                break
            if re.match(r"^\s*1\s+Scope\s*\n?$", line, re.IGNORECASE):
                if i + 1 < len(lines) and re.match(r"^\s*=+\s*\n?$", lines[i + 1]):
                    scope_index = i
                    break

        if scope_index is None:
            return None

        trimmed = []
        for line in lines[scope_index:]:
            clean_line = re.sub(r"^#+\s*", "", line).strip()
            if re.match(r"^Annex\s+[A-Z]\b", clean_line, re.IGNORECASE):
                break
            trimmed.append(line)

        return trimmed

    def _split_sections(self, lines: list[str]) -> list[dict]:
        """Parse lines into a hierarchical list of sections and subsections.

        Recognizes top-level sections (e.g., "4 ...") and one-level
        subsections (e.g., "4.1 ..."). Deeper headings (e.g., "4.1.1 ...")
        are skipped.

        Args:
            lines: Trimmed lines from the document.

        Returns:
            List of section dicts with keys:
                - ``title`` (str): Section heading text.
                - ``content`` (list[str]): Lines belonging directly to this section.
                - ``subsections`` (list[dict]): Nested subsection dicts with
                  ``title`` and ``content``.
        """
        sections: list[dict] = []
        current_section: dict | None = None
        current_subsection: dict | None = None

        section_pattern = re.compile(r"^\s*(\d+)\s+.+")
        subsection_pattern = re.compile(r"^\s*(\d+\.\d+)\s+.+")
        deeper_pattern = re.compile(r"^\s*(\d+\.\d+\.\d+)\s+.+")

        for raw_line in lines:
            line = raw_line.strip()
            line = re.sub(r"^#+\s*", "", line)

            # Skip decorative underline lines
            if re.match(r"^=+$", line) or re.match(r"^-+$", line):
                continue

            if deeper_pattern.match(line):
                continue  # Ignore headings deeper than level 2

            if subsection_pattern.match(line):
                current_subsection = {"title": line, "content": []}
                if current_section is not None:
                    current_section["subsections"].append(current_subsection)
                continue

            if section_pattern.match(line):
                current_section = {"title": line, "content": [], "subsections": []}
                sections.append(current_section)
                current_subsection = None
                continue

            # Append content lines to the current subsection or section
            if current_subsection is not None:
                current_subsection["content"].append(raw_line)
            elif current_section is not None:
                current_section["content"].append(raw_line)

        return sections

    def _remove_unwanted_sections(self, sections: list[dict]) -> list[str]:
        """Filter out Scope and References sections and collect text blocks.

        Args:
            sections: Parsed section list from ``_split_sections``.

        Returns:
            List of raw text strings from kept sections and their subsections.
        """
        cleaned_blocks = []

        for sec in sections:
            title_lower = sec["title"].lower()
            if "scope" in title_lower or "references" in title_lower:
                continue

            if sec["content"]:
                cleaned_blocks.append("".join(sec["content"]))

            for sub in sec["subsections"]:
                if sub["content"]:
                    cleaned_blocks.append("".join(sub["content"]))

        return cleaned_blocks

    def _normalize_paragraphs(self, text: str) -> str:
        """Normalize wrapped lines into single-line paragraphs.

        Strips blockquote markers, leading backslash dashes, and joins
        single-line breaks while preserving paragraph boundaries.

        Args:
            text: Raw section text.

        Returns:
            Paragraph-normalized text.
        """
        lines = text.splitlines()
        cleaned_lines = []

        for line in lines:
            line = line.lstrip()
            line = re.sub(r"^>\s*", "", line)
            line = re.sub(r"^\\\-\s*", "", line)
            cleaned_lines.append(line.rstrip())

        text = "\n".join(cleaned_lines)
        text = re.sub(r"\n{2,}", "<<<PARA>>>", text)
        text = re.sub(r"\n", " ", text)
        text = text.replace("<<<PARA>>>", "\n\n")

        return text

    def _remove_figures_tables(self, text: str) -> str:
        """Remove figure/table markup and associated caption lines.

        Removes:
            - Markdown image syntax for figures.
            - HTML tags.
            - Loose height attribute lines.
            - Figure and Table caption lines.
            - ASCII table borders and fixed-width columns.
            - Single and two-word orphan lines.

        Args:
            text: Paragraph-normalized text.

        Returns:
            Text with figures and tables stripped.
        """
        text = re.sub(r"!\[\]\(media/.*?\)\{.*?\}", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", "", text, flags=re.IGNORECASE)
        text = re.sub(r'^\s*height=".*?"\}\s*$', "", text, flags=re.MULTILINE)
        text = re.sub(
            r"^\s*(Figure|Table)\s+\d+.*$", "", text,
            flags=re.MULTILINE | re.IGNORECASE,
        )

        # Remove ASCII table rows
        lines = text.splitlines()
        cleaned: list[str] = []
        table_mode = False

        for line in lines:
            stripped = line.strip()
            if re.match(r"^[\-\+\| ]{5,}$", stripped):
                table_mode = True
                continue
            if re.search(r"\S\s{3,}\S", line):
                table_mode = True
                continue
            if table_mode:
                if stripped == "":
                    table_mode = False
                continue
            cleaned.append(line)

        text = "\n".join(cleaned)

        # Drop orphan lines with only 1-2 words
        final_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                final_lines.append(line)
                continue
            if len(stripped.split()) <= 2:
                continue
            final_lines.append(line)

        return "\n".join(final_lines)

    def _clean_spacing(self, text: str) -> str:
        """Normalize whitespace and remove lines with no alphanumeric content.

        Args:
            text: Text after figure/table removal.

        Returns:
            Whitespace-cleaned text.
        """
        lines = text.splitlines()
        cleaned = []

        for line in lines:
            line = line.replace("\t", " ")
            line = re.sub(r"[ ]{2,}", " ", line)
            stripped = line.strip()
            if stripped and not re.search(r"[A-Za-z0-9]", stripped):
                continue
            cleaned.append(line.rstrip())

        return "\n".join(cleaned)

    def _remove_references_and_md(self, text: str) -> str:
        """Remove inline citations, cross-references, and Markdown bold syntax.

        Args:
            text: Text after spacing cleanup.

        Returns:
            Text with references and Markdown markers stripped.
        """
        text = re.sub(r"\s*\\\?\[\d+(?:[,\-\s]+\d+)*\\\?\]\s*", "", text)
        text = re.sub(r"\s*\(\s*see[^)]+\)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        return text

    def _remove_duplicate_blocks(self, records: list[dict]) -> list[dict]:
        """Remove all records whose ``text`` appears more than once.

        If two or more records share identical ``text`` content, every
        occurrence is dropped rather than keeping one representative.

        Args:
            records: List of dicts with at least a ``"text"`` key.

        Returns:
            List with all duplicate-text records removed.
        """
        from collections import Counter

        counts = Counter(r["text"] for r in records)
        return [r for r in records if counts[r["text"]] == 1]

    def _final_filter_blocks(self, text: str, min_words: int = 40) -> str:
        """Filter out short paragraphs and sentences containing noise.

        Removes:
            - Paragraphs shorter than ``min_words`` words.
            - Sentences containing URLs or ``chr(`` calls.

        Args:
            text: Cleaned text.
            min_words: Minimum word count per paragraph to keep.

        Returns:
            Filtered multi-paragraph text.
        """
        blocks = text.split("\n\n")
        final_blocks = []

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            sentences = re.split(r'(?<=[.!?]) +', block)
            filtered_sentences = []

            for sent in sentences:
                if re.search(r'https?://|www\.', sent, re.IGNORECASE):
                    continue
                if "chr(" in sent:
                    continue
                filtered_sentences.append(sent.strip())

            cleaned_block = " ".join(filtered_sentences).strip()
            if len(cleaned_block.split()) > min_words:
                final_blocks.append(cleaned_block)

        return "\n\n".join(final_blocks)

    def _process_single_file(self, md_file: Path) -> tuple[str, list[str]] | None:
        """Process a single Markdown file and produce JSONL-ready strings.

        Expects the file to live under a directory whose name starts with
        "rel-" (case-insensitive) to derive the release label.

        Args:
            md_file: Path to a 3GPP Markdown specification file.

        Returns:
            A (release_name, list_of_json_strings) tuple, or None if the file
            cannot be processed (missing Scope section or release directory).
        """
        try:
            release_name = next(
                part for part in md_file.parts
                if part.lower().startswith("rel-")
            )
        except StopIteration:
            return None

        with md_file.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        trimmed = self._trim_by_scope_and_annex(lines)
        if trimmed is None:
            return None

        sections = self._split_sections(trimmed)
        cleaned_blocks = self._remove_unwanted_sections(sections)

        records = []
        for block in cleaned_blocks:
            block = self._normalize_paragraphs(block)
            block = self._remove_figures_tables(block)
            block = self._clean_spacing(block)
            block = self._remove_references_and_md(block)
            block = self._final_filter_blocks(block, min_words=40)

            clean_text = block.strip()
            if not clean_text:
                continue

            records.append({"text": clean_text, "file_name": md_file.name})

        # Drop every block whose text is duplicated (keep none of them).
        unique_records = self._remove_duplicate_blocks(records)

        results = [json.dumps(obj, ensure_ascii=False) for obj in unique_records]
        return release_name, results

    def run(self) -> None:
        """Process all 3GPP Markdown files and write results organised by release.

        Uses a process pool for parallel processing. Output files are named
        ``<release_name>.jsonl`` inside ``output_path``.
        """
        self.output_path.mkdir(parents=True, exist_ok=True)

        md_files = list(self.input_path.rglob("*.md"))
        max_workers = max(1, os.cpu_count() - 1)
        file_handles: dict[str, object] = {}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._process_single_file, f) for f in md_files]

            for future in as_completed(futures):
                result = future.result()
                if not result:
                    continue

                release_name, lines = result
                output_file = self.output_path / f"{release_name}.jsonl"

                if release_name not in file_handles:
                    file_handles[release_name] = open(
                        output_file, "a", encoding="utf-8"
                    )

                out = file_handles[release_name]
                for line in lines:
                    out.write(line + "\n")

        for fh in file_handles.values():
            fh.close()

        print(f"Done. Output written to: {self.output_path}")


if __name__ == "__main__":
    cleaner = ThreeGPPCleaner(
        input_dir="data/pretrain_data/raw/3GPP/",
        output_file="data/pretrain_data/processed/3GPP1/",
    )
    cleaner.run()