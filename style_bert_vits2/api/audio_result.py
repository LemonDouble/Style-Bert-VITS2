from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray


@dataclass
class AudioResult:
    """Generated audio data."""

    sr: int
    data: NDArray[np.int16]

    def save(self, path: Union[str, Path]) -> None:
        """Save as WAV file."""
        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sr)
            wf.writeframes(self.data.tobytes())

    def to_bytes(self) -> bytes:
        """Return WAV as bytes (for streaming/API responses)."""
        buf = io.BytesIO()
        with wave.open(buf, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sr)
            wf.writeframes(self.data.tobytes())
        return buf.getvalue()


class StyleAccessor:
    """Attribute-style access to style names. Supports IDE autocomplete."""

    def __init__(self, style2id: dict[str, int]) -> None:
        self._style2id = style2id
        for name in style2id:
            object.__setattr__(self, name, name)

    def __getattr__(self, name: str) -> str:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._style2id:
            return name
        available = list(self._style2id.keys())
        raise AttributeError(f"Style '{name}' not found. Available: {available}")

    def __dir__(self) -> list[str]:
        return list(self._style2id.keys()) + list(super().__dir__())

    def __iter__(self):
        return iter(self._style2id.keys())

    def __contains__(self, item: str) -> bool:
        return item in self._style2id

    def __repr__(self) -> str:
        return f"Styles({list(self._style2id.keys())})"
