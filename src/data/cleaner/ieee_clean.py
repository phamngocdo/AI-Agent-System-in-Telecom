import re
import json

from src.data.cleaner.base_cleaner import BaseCleaner


class IEEECleaner(BaseCleaner):
    """Clean IEEE Markdown documents and split them into section-level texts.

    Pipeline:
        1. Removes structural noise (watermark, headers, footers).
        2. Removes inline noise (URLs, cross-references).
        3. Normalizes PDF-style wrapped lines.
        4. Splits text into meaningful sections by numbered headings.
        5. Writes cleaned sections to an output JSONL file.

    Attributes:
        SECTION_PATTERN (re.Pattern): Regex matching numbered top-level
            headings such as "1 Introduction" or "4.1 Architecture Overview".
    """

    SECTION_PATTERN = re.compile(r"^\d+(\.\d+)*\s+[A-Z].+")

    def _clean_structure(self, text: str) -> str:
        """Remove document-level noise such as watermark and repeated header blocks.

        Removes:
            - IEEE license watermark.
            - Content before the first real section (Introduction/Overview).
            - Repeated IEEE header blocks.
            - Copyright footers.
            - Marketing slogan block at the end.

        Args:
            text: Raw Markdown text.

        Returns:
            Text with structural noise removed.
        """
        # Remove license watermark block
        text = re.sub(
            r"Authorized licensed use limited to:.*?Restrictions apply\.",
            "",
            text,
            flags=re.DOTALL,
        )

        # Trim content before first real section
        intro_match = re.search(r"\n\s*1\.?\s+Introduction\n", text)
        overview_match = re.search(r"\n\s*1\.?\s+Overview\n", text)

        start_positions = []
        if intro_match:
            start_positions.append(intro_match.start())
        if overview_match:
            start_positions.append(overview_match.start())

        if start_positions:
            text = text[min(start_positions):]

        # Remove repeated IEEE header blocks
        lines = text.splitlines()
        cleaned_lines = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("IEEE Std"):
                i += 1
                skip_count = 0
                while i < len(lines) and skip_count < 4:
                    next_line = lines[i].strip()
                    if next_line.startswith("IEEE Standard") or next_line == "":
                        i += 1
                        skip_count += 1
                    else:
                        break
                continue
            cleaned_lines.append(lines[i])
            i += 1
        text = "\n".join(cleaned_lines)

        # Remove copyright footer
        text = re.sub(
            r"\n?\s*\d+\s*\n\s*Copyright © \d{4} IEEE\. All rights reserved\.\s*",
            "\n",
            text,
        )
        text = re.sub(
            r"\n?\s*Copyright © \d{4} IEEE\. All rights reserved\.\s*",
            "\n",
            text,
        )

        # Remove marketing slogan block at the end
        marker = "RAISING THE"
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]

        return text

    def _clean_inline_noise(self, text: str) -> str:
        """Remove inline references and URLs.

        Removes:
            - Parenthetical cross-references such as "(see Section 3)".
            - External URLs.

        Args:
            text: Text after structural cleaning.

        Returns:
            Text with inline noise removed.
        """
        text = re.sub(r'\((?i:see)[^)]*\)', '', text)
        text = re.sub(r'https?://[^\s)]+', '', text)
        return text

    def _normalize_wrapped_lines(self, text: str) -> str:
        """Normalize PDF-style wrapped lines.

        Converts single line breaks into spaces while preserving
        paragraph boundaries (double newlines).

        Args:
            text: Text after noise removal.

        Returns:
            Text with normalized line wrapping.
        """
        text = text.replace(".;", ".")

        # Preserve paragraph breaks using placeholder
        text = text.replace("\n\n", "<<<PARA>>>")
        text = re.sub(r"\n", " ", text)
        text = text.replace("<<<PARA>>>", "\n\n")

        # Collapse extra spaces
        text = re.sub(r"[ \t]{2,}", " ", text)

        return text.strip()

    def _is_meaningful(self, text: str, min_words: int = 40) -> bool:
        """Check whether a section has sufficient content.

        Args:
            text: Section text.
            min_words: Minimum number of alphanumeric words required.

        Returns:
            True if the section meets the minimum word threshold.
        """
        words = re.findall(r"\b[a-zA-Z0-9]+\b", text)
        return len(words) >= min_words

    def _split_by_section(self, text: str) -> list[str]:
        """Split cleaned text into section-level blocks.

        Headings are identified by ``SECTION_PATTERN``. Only sections
        that meet the minimum word count are kept.

        Args:
            text: Fully cleaned text.

        Returns:
            List of meaningful section text strings.
        """
        lines = text.split("\n")
        sections = []
        current_section: list[str] = []

        for line in lines:
            stripped = line.strip()

            if self.SECTION_PATTERN.match(stripped):
                # Save the completed section before starting a new one
                if current_section:
                    section_text = "\n".join(current_section).strip()
                    if section_text and self._is_meaningful(section_text):
                        sections.append(section_text)
                current_section = []
                continue

            if stripped:
                current_section.append(stripped)

        # Append the last section
        if current_section:
            section_text = "\n".join(current_section).strip()
            if section_text and self._is_meaningful(section_text):
                sections.append(section_text)

        return sections

    def _process_text(self, text: str) -> list[str]:
        """Run the full per-document cleaning pipeline.

        Args:
            text: Raw Markdown text from one file.

        Returns:
            List of cleaned section-level text strings.
        """
        text = self._clean_structure(text)
        text = self._clean_inline_noise(text)
        text = self._normalize_wrapped_lines(text)
        return self._split_by_section(text)

    def run(self) -> None:
        """Process all Markdown files in the input directory and write to JSONL.

        Each output record contains:
            ``{"text": section_text, "file_name": source_file_name}``
        """
        md_files = list(self.input_path.rglob("*.md"))
        print(f"Found {len(md_files)} .md files")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        total_sections = 0

        with open(self.output_path, "w", encoding="utf-8") as out_f:
            for file_path in md_files:
                text = file_path.read_text(encoding="utf-8")
                sections = self._process_text(text)

                for section_text in sections:
                    record = {
                        "text": section_text,
                        "file_name": file_path.name,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_sections += 1

        print(f"Done. Total sections saved: {total_sections}")


if __name__ == "__main__":
    cleaner = IEEECleaner(
        input_dir="data/pretrain_data/raw/IEEE",
        output_file="data/pretrain_data/processed/IEEE/IEEE.jsonl",
    )
    cleaner.run()