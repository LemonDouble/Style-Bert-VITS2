from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from style_bert_vits2.api.resources import load_bert, prepare_bert
from style_bert_vits2.api.speaker import Speaker
from style_bert_vits2.logging import logger
from style_bert_vits2.utils.strenum import StrEnum


class Lang(StrEnum):
    """Language enum for the public API."""

    JA = "JP"  # Internal code uses "JP"


class TTS:
    """Style-Bert-VITS2 inference engine.

    Manages shared resources (BERT) and loaded speakers.

    Usage:
        tts = TTS(device="cuda")
        elaina = tts.load("elaina", "./models/Elaina")
        audio = elaina.generate("先生、大丈夫ですか？", lang=Lang.JA)
        audio.save("output.wav")
    """

    def __init__(
        self,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ) -> None:
        """Initialize TTS engine. Downloads and loads BERT if needed.

        Args:
            device: PyTorch device ("cpu", "cuda", "cuda:0", etc.)
            cache_dir: Custom cache directory for model downloads.
        """
        self._device = device
        self._cache_dir = cache_dir
        self._speakers: dict[str, Speaker] = {}

        load_bert(device=device, cache_dir=cache_dir)
        logger.info(f"TTS ready on {device}")

    @staticmethod
    def prepare(cache_dir: Optional[str] = None) -> None:
        """Download shared resources only. No GPU needed.

        Use this in Docker builds or CI to pre-cache models:
            RUN python -c "from style_bert_vits2 import TTS; TTS.prepare()"
        """
        prepare_bert(cache_dir=cache_dir)

    def load(self, name: str, model_dir: Union[str, Path]) -> Speaker:
        """Load a fine-tuned speaker model.

        Args:
            name: Speaker name (for identification).
            model_dir: Path to directory containing:
                - *.safetensors (model weights)
                - config.json (hyperparameters)
                - style_vectors.npy (style vectors)

        Returns:
            Speaker instance with .generate() method.
        """
        speaker = Speaker(name=name, model_dir=Path(model_dir), device=self._device)
        self._speakers[name] = speaker
        return speaker

    @property
    def speakers(self) -> dict[str, Speaker]:
        """Currently loaded speakers."""
        return dict(self._speakers)

    def __repr__(self) -> str:
        names = list(self._speakers.keys())
        return f"TTS(device='{self._device}', speakers={names})"
