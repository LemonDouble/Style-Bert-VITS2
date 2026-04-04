"""ONNX Runtime inference for Style-Bert-VITS2.

Replaces PyTorch inference with ONNX Runtime for CPU speed optimization.
Expects two ONNX models: bert.onnx and synthesizer.onnx.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from style_bert_vits2.constants import Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.models import commons
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.nlp import (
    clean_text_with_given_phone_tone,
    cleaned_text_to_sequence,
)
from style_bert_vits2.nlp.japanese.g2p import text_to_sep_kata
from style_bert_vits2.nlp.symbols import SYMBOLS


def _get_tokenizer():
    """Get the correct BertJapaneseTokenizer (not the broken fast tokenizer)."""
    from style_bert_vits2.nlp import bert_models

    if not bert_models.is_tokenizer_loaded():
        from style_bert_vits2.constants import BERT_JP_REPO
        bert_models.load_tokenizer(pretrained_model_name_or_path=BERT_JP_REPO)
    return bert_models.load_tokenizer()


def extract_bert_feature_onnx(
    text: str,
    word2ph: list[int],
    onnx_session,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
) -> NDArray:
    """Extract BERT features using ONNX Runtime session.

    Mirrors the logic in nlp/japanese/bert_feature.py but uses ONNX.
    """
    # Same preprocessing as PyTorch version
    text = "".join(text_to_sep_kata(text, raise_yomi_error=False)[0])
    if assist_text:
        assist_text = "".join(text_to_sep_kata(assist_text, raise_yomi_error=False)[0])

    tokenizer = _get_tokenizer()
    inputs = tokenizer(text, return_tensors="np")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    res = onnx_session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    })[0][0]  # [seq_len, hidden_dim]

    style_res_mean = None
    if assist_text:
        style_inputs = tokenizer(assist_text, return_tensors="np")
        style_res = onnx_session.run(None, {
            "input_ids": style_inputs["input_ids"],
            "attention_mask": style_inputs["attention_mask"],
        })[0][0]
        style_res_mean = style_res.mean(axis=0)

    assert len(word2ph) == len(text) + 2, text
    phone_level_feature = []
    for i in range(len(word2ph)):
        if assist_text and style_res_mean is not None:
            repeat_feature = (
                np.tile(res[i], (word2ph[i], 1)) * (1 - assist_text_weight)
                + np.tile(style_res_mean, (word2ph[i], 1)) * assist_text_weight
            )
        else:
            repeat_feature = np.tile(res[i], (word2ph[i], 1))
        phone_level_feature.append(repeat_feature)

    phone_level_feature = np.concatenate(phone_level_feature, axis=0)
    return phone_level_feature.T  # [hidden_dim, phone_len]


def get_text_onnx(
    text: str,
    hps: HyperParameters,
    bert_session,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Text preprocessing + BERT feature extraction via ONNX."""
    use_jp_extra = hps.version.endswith("JP-Extra")
    norm_text, phone, tone, word2ph = clean_text_with_given_phone_tone(
        text,
        Languages.JP,
        given_phone=given_phone,
        given_tone=given_tone,
        use_jp_extra=use_jp_extra,
        raise_yomi_error=False,
    )
    phone, tone, language = cleaned_text_to_sequence(phone, tone, Languages.JP)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    ja_bert = extract_bert_feature_onnx(
        norm_text,
        word2ph,
        bert_session,
        assist_text,
        assist_text_weight,
    )
    del word2ph
    assert ja_bert.shape[-1] == len(phone), phone

    phone = np.array(phone, dtype=np.int64)
    tone = np.array(tone, dtype=np.int64)
    language = np.array(language, dtype=np.int64)
    return ja_bert, phone, tone, language


def infer_onnx(
    text: str,
    style_vec: NDArray[Any],
    sdp_ratio: float,
    noise_scale: float,
    noise_scale_w: float,
    length_scale: float,
    sid: int,
    language: Languages,
    hps: HyperParameters,
    bert_session,
    synth_session,
    skip_start: bool = False,
    skip_end: bool = False,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
) -> NDArray[Any]:
    """Full ONNX inference pipeline: text → BERT → Synthesizer → audio."""
    ja_bert, phones, tones, lang_ids = get_text_onnx(
        text,
        hps,
        bert_session,
        assist_text=assist_text,
        assist_text_weight=assist_text_weight,
        given_phone=given_phone,
        given_tone=given_tone,
    )

    if skip_start:
        phones = phones[3:]
        tones = tones[3:]
        lang_ids = lang_ids[3:]
        ja_bert = ja_bert[:, 3:]
    if skip_end:
        phones = phones[:-2]
        tones = tones[:-2]
        lang_ids = lang_ids[:-2]
        ja_bert = ja_bert[:, :-2]

    # Add batch dimension
    x = phones[np.newaxis, :]                       # [1, phone_len]
    x_lengths = np.array([phones.shape[0]], dtype=np.int64)  # [1]
    t = tones[np.newaxis, :]                         # [1, phone_len]
    l = lang_ids[np.newaxis, :]                      # [1, phone_len]
    b = ja_bert[np.newaxis, :, :]                    # [1, 1024, phone_len]
    s = style_vec[np.newaxis, :].astype(np.float32)  # [1, 256]
    sid_arr = np.array([sid], dtype=np.int64)        # [1]

    output = synth_session.run(None, {
        "x": x,
        "x_lengths": x_lengths,
        "sid": sid_arr,
        "tone": t,
        "language": l,
        "bert": b.astype(np.float32),
        "style_vec": s,
        "noise_scale": np.array([noise_scale], dtype=np.float32),
        "length_scale": np.array([length_scale], dtype=np.float32),
        "noise_scale_w": np.array([noise_scale_w], dtype=np.float32),
        "sdp_ratio": np.array([sdp_ratio], dtype=np.float32),
    })

    audio = output[0][0, 0]  # [audio_len]
    return audio
