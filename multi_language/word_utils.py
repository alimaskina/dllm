"""Word segmentation helpers for multilingual tokenization audit."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterator

# Languages where whitespace-delimited tokens approximate linguistic words.
WHITESPACE_LANGS = frozenset({"en", "de", "ru", "tr", "fi", "ko"})

# CJK: segment into contiguous Han blocks (minimal PoC without jieba/mecab).
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]+")
# Korean Hangul syllable blocks (each block ≈ one morpheme/word in PoC).
_HANGUL_RE = re.compile(r"[\uac00-\ud7af]+")
# Latin/Cyrillic/etc. alphabetic runs inside whitespace token.
_ALPHA_RE = re.compile(r"[\w']+", re.UNICODE)


def _is_punct_or_symbol(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S")


def segment_zh(text: str) -> list[str]:
    """Split Chinese text into Han character sequences (one char = one 'word' if isolated)."""
    words: list[str] = []
    for m in _CJK_RE.finditer(text):
        chunk = m.group()
        # Treat each Han character as a lexical unit (conservative fragmentation proxy).
        words.extend(list(chunk))
    return words


def segment_ko(text: str) -> list[str]:
    """Korean: Hangul syllable blocks + whitespace tokens for mixed text."""
    words: list[str] = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        m = _HANGUL_RE.match(text, i)
        if m:
            # Each Hangul syllable block is one word unit.
            words.extend(list(m.group()))
            i = m.end()
            continue
        # Latin/digits/punct clumps
        j = i
        while j < len(text) and not text[j].isspace() and not _HANGUL_RE.match(text, j):
            j += 1
        tok = text[i:j].strip()
        if tok and not all(_is_punct_or_symbol(c) for c in tok):
            words.append(tok)
        i = j
    return words


def segment_whitespace(text: str) -> list[str]:
    """Standard whitespace word tokenization."""
    return [w for w in re.findall(r"\S+", text) if w]


def segment_words(text: str, lang: str) -> list[str]:
    """Return linguistic word units for a language code (ISO 639-1)."""
    if lang == "zh":
        return segment_zh(text)
    if lang == "ko":
        return segment_ko(text)
    if lang in WHITESPACE_LANGS:
        return segment_whitespace(text)
    return segment_whitespace(text)


def word_char_len(word: str) -> int:
    return len(word)


def iter_corpus_words(texts: Iterator[str], lang: str) -> Iterator[tuple[str, int]]:
    """Yield (word, char_len) from a stream of documents."""
    for text in texts:
        if not text or not text.strip():
            continue
        for word in segment_words(text, lang):
            core = word.strip()
            if len(core) < 1:
                continue
            if all(_is_punct_or_symbol(c) for c in core):
                continue
            yield core, word_char_len(core)
