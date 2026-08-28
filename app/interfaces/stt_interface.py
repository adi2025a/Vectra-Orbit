from abc import ABC, abstractmethod
from typing import Optional

class BaseSTT(ABC):
    """Abstract Interface for Speech-To-Text (STT) Transcription."""

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000, language: Optional[str] = "en") -> str:
        """
        Transcribe complete audio segment bytes (PCM 16-bit) to text.
        Returns the recognized transcript string.
        """
        pass
