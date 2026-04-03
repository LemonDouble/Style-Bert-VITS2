from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from numpy.typing import NDArray

from style_bert_vits2.api.audio_result import AudioResult, StyleAccessor
from style_bert_vits2.constants import Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.voice import adjust_voice


if TYPE_CHECKING:
    from style_bert_vits2.models.models_jp_extra import (
        SynthesizerTrn as SynthesizerTrnJPExtra,
    )


class Speaker:
    """A loaded fine-tuned speaker model. Created via TTS.load()."""

    def __init__(self, name: str, model_dir: Path, device: str) -> None:
        self.name = name
        self._device = device
        self._model_dir = model_dir

        # Find newest safetensors
        safetensors_files = sorted(
            model_dir.glob("*.safetensors"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if not safetensors_files:
            raise FileNotFoundError(f"No .safetensors file in {model_dir}")

        self._model_path = safetensors_files[0]
        self._config_path = model_dir / "config.json"
        self._style_vec_path = model_dir / "style_vectors.npy"

        if not self._config_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_dir}")
        if not self._style_vec_path.exists():
            raise FileNotFoundError(f"style_vectors.npy not found in {model_dir}")

        # Load config + style vectors
        self._hps = HyperParameters.load_from_json(self._config_path)
        self._style_vectors: NDArray = np.load(self._style_vec_path)

        if hasattr(self._hps.data, "style2id"):
            self._style2id: dict[str, int] = self._hps.data.style2id
        else:
            num_styles = self._hps.data.num_styles
            self._style2id = {str(i): i for i in range(num_styles)}

        self.styles = StyleAccessor(self._style2id)

        # Lazy-loaded
        self._net_g: Optional[SynthesizerTrnJPExtra] = None

        logger.info(
            f"Speaker '{name}' loaded from {self._model_path.name} "
            f"(styles: {list(self._style2id.keys())})"
        )

    def _ensure_model(self) -> SynthesizerTrnJPExtra:
        if self._net_g is not None:
            return self._net_g

        from style_bert_vits2.models.infer import get_net_g

        self._net_g = get_net_g(
            model_path=str(self._model_path),
            version=self._hps.version,
            device=self._device,
            hps=self._hps,
        )
        return self._net_g

    def _get_style_vector(self, style: str, weight: float) -> NDArray:
        style_id = self._style2id.get(style)
        if style_id is None:
            available = list(self._style2id.keys())
            raise ValueError(f"Style '{style}' not found. Available: {available}")
        mean = self._style_vectors[0]
        vec = self._style_vectors[style_id]
        return mean + (vec - mean) * weight

    def generate(
        self,
        text: str,
        *,
        lang: Union[str, Languages] = Languages.JP,
        style: str = "Neutral",
        speaker_id: int = 0,
        speed: float = 1.0,
        sdp_ratio: float = 0.2,
        noise: float = 0.6,
        noise_w: float = 0.8,
        pitch_scale: float = 1.0,
        intonation_scale: float = 1.0,
        style_weight: float = 1.0,
    ) -> AudioResult:
        """Generate speech from text.

        Args:
            text: Text to synthesize.
            lang: Language (Lang.JA or Lang.EN).
            style: Style name (e.g. "Neutral", "Happy").
            speaker_id: Speaker ID for multi-speaker models.
            speed: Speech speed. 1.0 = normal, <1.0 = faster, >1.0 = slower.
            sdp_ratio: SDP/DP ratio. 0=DP only, 1=SDP only.
            noise: Noise scale for DP.
            noise_w: Noise scale for SDP.
            pitch_scale: Pitch adjustment (1.0 = no change).
            intonation_scale: Intonation adjustment (1.0 = no change).
            style_weight: Style vector weight.

        Returns:
            AudioResult with .save() and .to_bytes() methods.
        """
        import torch

        from style_bert_vits2.models.infer import infer

        # Resolve lang string to Languages enum
        lang_str = Languages(lang.value if hasattr(lang, "value") else str(lang))

        net_g = self._ensure_model()
        style_vec = self._get_style_vector(style, style_weight)

        with torch.no_grad():
            audio = infer(
                text=text,
                sdp_ratio=sdp_ratio,
                noise_scale=noise,
                noise_scale_w=noise_w,
                length_scale=speed,
                sid=speaker_id,
                language=lang_str,
                hps=self._hps,
                net_g=net_g,
                device=self._device,
                style_vec=style_vec,
            )

        if pitch_scale != 1.0 or intonation_scale != 1.0:
            _, audio = adjust_voice(
                fs=self._hps.data.sampling_rate,
                wave=audio,
                pitch_scale=pitch_scale,
                intonation_scale=intonation_scale,
            )

        # Convert to 16-bit PCM
        audio = audio / np.abs(audio).max()
        audio = (audio * 32767).astype(np.int16)

        return AudioResult(sr=self._hps.data.sampling_rate, data=audio)

    def __repr__(self) -> str:
        return f"Speaker('{self.name}', styles={list(self._style2id.keys())})"
