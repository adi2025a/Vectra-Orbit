from abc import ABC, abstractmethod

class BaseVAD(ABC):
    """Abstract Interface for Voice Activity Detection (VAD)."""

    @abstractmethod
    def process_chunk(self, audio_bytes: bytes, sample_rate: int = 16000) -> bool:
        """
        Process a chunk of audio bytes (PCM 16-bit) and return True if human speech is detected.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state/buffer of the VAD detector."""
        pass
