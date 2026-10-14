"""Recursive character chunking that keeps page/slide numbers for citations."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .extract import Section

# Try to break on paragraphs first, then lines, sentences, and finally words.
_SEPARATORS = ("\n\n", "\n", ". ", " ")


@dataclass
class Chunk:
    text: str
    page: int | None
    index: int


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_sections(sections: list[Section], size: int, overlap: int) -> list[Chunk]:
    """Split each section into pieces of at most `size` chars, then prepend up to `overlap`
    chars from the previous piece of the same section so context carries across boundaries."""
    chunks: list[Chunk] = []
    for section in sections:
        text = normalize(section.text)
        if not text:
            continue
        pieces = [p.strip() for p in _split(text, size, _SEPARATORS) if p.strip()]
        for i, piece in enumerate(pieces):
            if i > 0 and overlap > 0:
                piece = f"{_tail(pieces[i - 1], overlap)} {piece}".strip()
            chunks.append(Chunk(piece, section.page, len(chunks)))
    return chunks


def _split(text: str, size: int, separators: tuple[str, ...]) -> list[str]:
    if len(text) <= size:
        return [text]
    for i, sep in enumerate(separators):
        if sep not in text:
            continue
        out: list[str] = []
        buf = ""
        for part in text.split(sep):
            candidate = f"{buf}{sep}{part}" if buf else part
            if len(candidate) <= size:
                buf = candidate
                continue
            if buf:
                out.append(buf)
            if len(part) > size:
                out.extend(_split(part, size, separators[i + 1 :]))
                buf = ""
            else:
                buf = part
        if buf:
            out.append(buf)
        return out
    return [text[start : start + size] for start in range(0, len(text), size)]


def _tail(text: str, length: int) -> str:
    if len(text) <= length:
        return text
    tail = text[-length:]
    space = tail.find(" ")
    return tail[space + 1 :] if 0 <= space < length // 2 else tail
