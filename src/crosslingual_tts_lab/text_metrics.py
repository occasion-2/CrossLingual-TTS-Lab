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


class ASRTextAdapter:
    def normalize(self, text: str) -> str:
        raise NotImplementedError


class DefaultASRAdapter(ASRTextAdapter):
    """Fallback adapter: lowercase, strip basic punctuation, and normalize whitespace."""
    def normalize(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        # Remove standard punctuation and symbols, keeping only alphanumeric and whitespace
        text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
        return " ".join(text.split())


class EnglishASRAdapter(ASRTextAdapter):
    """English ASR adapter: lowercase, remove punctuation except apostrophes, normalize whitespace."""
    def normalize(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        # Keep letters, numbers, spaces, and apostrophes
        text = re.sub(r"[^\w\s']", "", text, flags=re.UNICODE)
        return " ".join(text.split())


class RussianASRAdapter(ASRTextAdapter):
    """Russian ASR adapter: lowercase, replace ё with е, remove punctuation, normalize whitespace."""
    def normalize(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        text = text.replace("ё", "е")
        text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
        return " ".join(text.split())


class ChineseASRAdapter(ASRTextAdapter):
    """Chinese ASR adapter: lowercase, remove all spaces, remove all punctuation."""
    def normalize(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        # Remove all spaces and punctuation
        text = re.sub(r"[^\w]", "", text, flags=re.UNICODE)
        text = text.replace("_", "")
        return text


class JapaneseASRAdapter(ChineseASRAdapter):
    pass


class KoreanASRAdapter(ASRTextAdapter):
    """Korean ASR adapter: lowercase, keep spaces, remove punctuation."""
    def normalize(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
        return " ".join(text.split())


_ASR_ADAPTER_REGISTRY: dict[str, ASRTextAdapter] = {
    "en": EnglishASRAdapter(),
    "ru": RussianASRAdapter(),
    "zh": ChineseASRAdapter(),
    "cmn": ChineseASRAdapter(),
    "ja": JapaneseASRAdapter(),
    "ko": KoreanASRAdapter(),
}


def get_asr_adapter(language: str) -> ASRTextAdapter:
    lang_key = language.lower().replace("_", "-").split("-")[0]
    return _ASR_ADAPTER_REGISTRY.get(lang_key, DefaultASRAdapter())


def choose_error_rate(language: str, reference: str, hypothesis: str) -> tuple[str, float]:
    adapter = get_asr_adapter(language)
    norm_ref = adapter.normalize(reference)
    norm_hyp = adapter.normalize(hypothesis)

    lang_key = language.lower().replace("_", "-").split("-")[0]
    if lang_key in {"zh", "cmn", "ja", "ko"}:
        return "cer", char_error_rate(norm_ref, norm_hyp)
    return "wer", word_error_rate(norm_ref, norm_hyp)


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
