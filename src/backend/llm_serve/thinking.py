import re
from collections.abc import Callable


THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)
THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    if not text:
        return ""

    close_matches = list(THINK_CLOSE_RE.finditer(text))
    if close_matches:
        text = text[close_matches[-1].end():]

    text = THINK_BLOCK_RE.sub("", text)
    open_match = THINK_OPEN_RE.search(text)
    if open_match:
        text = text[:open_match.start()]

    text = THINK_CLOSE_RE.sub("", text)
    return text.strip()


class ThinkingStreamFilter:
    _OPEN_START = "<think"
    _CLOSE_START = "</think"

    def __init__(
        self,
        strip_text: Callable[[str], str] = strip_thinking,
    ) -> None:
        self.strip_text = strip_text
        self.buffer = ""
        self.in_think = False
        self.has_visible_text = False

    def feed(self, text: str) -> list[str]:
        if not text:
            return []

        self.buffer += text
        visible_chunks: list[str] = []

        while self.buffer:
            if self.in_think:
                close_match = THINK_CLOSE_RE.search(self.buffer)
                if close_match is None:
                    self.buffer = self._keep_possible_close_suffix(self.buffer)
                    break

                self.buffer = self.buffer[close_match.end():]
                self.in_think = False
                continue

            open_match = THINK_OPEN_RE.search(self.buffer)
            close_match = THINK_CLOSE_RE.search(self.buffer)

            if close_match is not None and (
                open_match is None or close_match.start() < open_match.start()
            ):
                self._append_visible(visible_chunks, self.buffer[:close_match.start()])
                self.buffer = self.buffer[close_match.end():]
                continue

            if open_match is not None:
                self._append_visible(visible_chunks, self.buffer[:open_match.start()])
                self.buffer = self.buffer[open_match.end():]
                self.in_think = True
                continue

            visible_prefix, self.buffer = self._split_visible_prefix(self.buffer)
            self._append_visible(visible_chunks, visible_prefix)
            break

        return visible_chunks

    def flush(self) -> list[str]:
        if self.in_think:
            self.buffer = ""
            return []

        visible_text = self.strip_text(self.buffer)
        self.buffer = ""
        if not visible_text:
            return []

        if not self.has_visible_text:
            visible_text = visible_text.lstrip()
        self.has_visible_text = True
        return [visible_text] if visible_text else []

    def _append_visible(self, chunks: list[str], text: str) -> None:
        if not text:
            return

        if not self.has_visible_text:
            text = text.lstrip()
        if not text:
            return

        self.has_visible_text = True
        chunks.append(text)

    def _split_visible_prefix(self, text: str) -> tuple[str, str]:
        lower_text = text.lower()
        last_tag_start = lower_text.rfind("<")

        if last_tag_start != -1:
            suffix = lower_text[last_tag_start:]
            if (
                self._OPEN_START.startswith(suffix)
                or suffix.startswith(self._OPEN_START)
                or self._CLOSE_START.startswith(suffix)
                or suffix.startswith(self._CLOSE_START)
            ):
                return text[:last_tag_start], text[last_tag_start:]

        return text, ""

    def _keep_possible_close_suffix(self, text: str) -> str:
        lower_text = text.lower()
        last_tag_start = lower_text.rfind("<")

        if last_tag_start != -1:
            suffix = lower_text[last_tag_start:]
            if self._CLOSE_START.startswith(suffix) or suffix.startswith(self._CLOSE_START):
                return text[last_tag_start:]

        return ""
