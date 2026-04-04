from __future__ import annotations

from typing import Optional

from style_bert_vits2.constants import BERT_JP_REPO
from style_bert_vits2.logging import logger
from style_bert_vits2.nlp import bert_models


def prepare_bert(cache_dir: Optional[str] = None) -> None:
    """Download JP BERT model and tokenizer to local cache. No GPU needed."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    logger.info(f"Downloading JP BERT from {BERT_JP_REPO}...")
    AutoModelForMaskedLM.from_pretrained(BERT_JP_REPO, cache_dir=cache_dir)
    AutoTokenizer.from_pretrained(BERT_JP_REPO, cache_dir=cache_dir)
    logger.info("JP BERT download complete.")


def load_bert(device: str, cache_dir: Optional[str] = None) -> None:
    """Download (if needed) + load JP BERT model and tokenizer into memory."""
    if bert_models.is_model_loaded() and bert_models.is_tokenizer_loaded():
        bert_models.transfer_model(device)
        return

    bert_models.load_model(
        pretrained_model_name_or_path=BERT_JP_REPO,
        device_map=device,
        cache_dir=cache_dir,
    )
    bert_models.load_tokenizer(
        pretrained_model_name_or_path=BERT_JP_REPO,
        cache_dir=cache_dir,
    )
