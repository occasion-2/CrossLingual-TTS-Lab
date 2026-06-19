from __future__ import annotations

import re


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_tokens = _word_tokens(reference)
    hyp_tokens = _word_tokens(hypothesis)
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return _edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def char_error_rate(reference: str, hypothesis: str) -> float:
    ref_chars = _chars(reference)
    hyp_chars = _chars(hypothesis)
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return _edit_distance(ref_chars, hyp_chars) / len(ref_chars)


def choose_error_rate(language: str, reference: str, hypothesis: str) -> tuple[str, float]:
    if language.lower() in {"zh", "ja", "ko"}:
        return "cer", char_error_rate(reference, hypothesis)
    return "wer", word_error_rate(reference, hypothesis)


def _word_tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORD_RE.finditer(text)]


def _chars(text: str) -> list[str]:
    return [char.casefold() for char in text if not char.isspace()]


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            cost = 0 if left_item == right_item else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]
