"""Shared filters for multi-token word unmasking experiments."""

from __future__ import annotations

import re

INFOBOX_JUNK = re.compile(
    r"(target:|Question:|Answer:|answers:|-lrb-|-rrb-|@-,@|"
    r"image;|date;|place;|spouse;|genre;|website\.|awards;|nationality;|active;|"
    r"name;|caption;|dynasty;|birth_|death_|occupation;|children;|father;|mother;|"
    r"Options:|Choices:|from:|origin;|background;|label;)",
    re.I,
)

CONTRACTION_FRAGMENTS = frozenset({"n't", "'s", "'re", "'ve", "'ll", "'d", "'m", "'t"})

_ALPHA_CORE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def word_core(word: str) -> str:
    return re.sub(r"^[^\w']+|[^\w']+$", "", word)


def is_alpha_multi(word: str) -> bool:
    """Tier A: alphabetic core (>=3 chars), incl. glued punctuation like ``present.``."""
    core = word_core(word)
    return bool(_ALPHA_CORE.fullmatch(core)) and len(core) >= 3


def is_lexical_loose(word: str) -> bool:
    """Legacy Tier B: letters-only core and no infobox/task junk; allows glued punctuation."""
    if INFOBOX_JUNK.search(word):
        return False
    core = word_core(word)
    return bool(_ALPHA_CORE.fullmatch(core)) and len(core) >= 3


def is_lexical(word: str) -> bool:
    """Default experiment tier: true multi-token words, not word+punctuation artifacts."""
    if not is_lexical_loose(word):
        return False
    core = word_core(word)
    if word != core:
        return False
    if not core or not core[0].isalpha():
        return False
    if core in CONTRACTION_FRAGMENTS:
        return False
    return True


def filter_instances(instances, *, tier: str = "lexical"):
    """Filter WordInstance list by word tier."""
    if tier == "none" or tier == "all":
        return instances
    if tier == "alpha":
        pred = is_alpha_multi
    elif tier == "lexical_loose":
        pred = is_lexical_loose
    elif tier == "lexical":
        pred = is_lexical
    else:
        raise ValueError(f"unknown tier: {tier}")
    return [inst for inst in instances if pred(inst.word)]
