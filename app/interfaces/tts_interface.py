from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional

class BaseTTS(ABC):
    """Abstract Interface for Text-To-Speech (TTS) Synthesis."""

    @abstractmethod
    def synthesize_stream(
        self, 
        text_stream: AsyncGenerator[str, None], 
        voice: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream audio bytes (PCM/mp3) as text chunks arrive from the LLM.
        """
        pass

    @abstractmethod
    async def synthesize_text(
        self, 
        text: str, 
        voice: Optional[str] = None
    ) -> bytes:
        """
        Synthesize complete static text to audio bytes.
        """
        pass
