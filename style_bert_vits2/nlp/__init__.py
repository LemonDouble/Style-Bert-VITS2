from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp.symbols import (
    LANGUAGE_ID_MAP,
    LANGUAGE_TONE_START_MAP,
    SYMBOLS,
)


if TYPE_CHECKING:
    import torch


__symbol_to_id = {s: i for i, s in enumerate(SYMBOLS)}


def extract_bert_feature(
    text: str,
    word2ph: list[int],
    language: Languages,
    device: str,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
) -> torch.Tensor:
    if language != Languages.JP:
        raise ValueError(f"Only JP is supported, got: {language}")

    from style_bert_vits2.nlp.japanese.bert_feature import extract_bert_feature

    return extract_bert_feature(text, word2ph, device, assist_text, assist_text_weight)


def clean_text(
    text: str,
    language: Languages,
    use_jp_extra: bool = True,
    raise_yomi_error: bool = False,
) -> tuple[str, list[str], list[int], list[int]]:
    if language != Languages.JP:
        raise ValueError(f"Only JP is supported, got: {language}")

    from style_bert_vits2.nlp.japanese.g2p import g2p
    from style_bert_vits2.nlp.japanese.normalizer import normalize_text

    norm_text = normalize_text(text)
    phones, tones, word2ph = g2p(norm_text, use_jp_extra, raise_yomi_error)

    return norm_text, phones, tones, word2ph


def clean_text_with_given_phone_tone(
    text: str,
    language: Languages,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
    use_jp_extra: bool = True,
    raise_yomi_error: bool = False,
) -> tuple[str, list[str], list[int], list[int]]:
    norm_text, phone, tone, word2ph = clean_text(
        text,
        language,
        use_jp_extra=use_jp_extra,
        raise_yomi_error=raise_yomi_error,
    )

    if given_phone is not None and given_tone is not None:
        if len(given_phone) != len(given_tone):
            raise InvalidPhoneError(
                f"Length of given_phone ({len(given_phone)}) != length of given_tone ({len(given_tone)})"
            )
        if len(given_phone) != sum(word2ph):
            from style_bert_vits2.nlp.japanese.g2p import adjust_word2ph

            if not use_jp_extra:
                given_phone = [p if p != "N" else "n" for p in given_phone]
            word2ph = adjust_word2ph(word2ph, phone, given_phone)
            if len(given_phone) != sum(word2ph):
                raise InvalidPhoneError(
                    f"Length of given_phone ({len(given_phone)}) != sum of word2ph ({sum(word2ph)})"
                )
        phone = given_phone
        if len(phone) != len(given_tone):
            raise InvalidToneError(
                f"Length of phone ({len(phone)}) != length of given_tone ({len(given_tone)})"
            )
        tone = given_tone

    elif given_tone is not None:
        if len(phone) != len(given_tone):
            raise InvalidToneError(
                f"Length of phone ({len(phone)}) != length of given_tone ({len(given_tone)})"
            )
        tone = given_tone

    return norm_text, phone, tone, word2ph


def cleaned_text_to_sequence(
    cleaned_phones: list[str], tones: list[int], language: Languages
) -> tuple[list[int], list[int], list[int]]:
    phones = [__symbol_to_id[symbol] for symbol in cleaned_phones]
    tone_start = LANGUAGE_TONE_START_MAP[language]
    tones = [i + tone_start for i in tones]
    lang_id = LANGUAGE_ID_MAP[language]
    lang_ids = [lang_id for i in phones]

    return phones, tones, lang_ids


class InvalidPhoneError(ValueError):
    pass


class InvalidToneError(ValueError):
    pass
