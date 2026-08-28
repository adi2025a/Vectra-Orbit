import io
import edge_tts
from typing import AsyncGenerator, Optional
from app.interfaces.tts_interface import BaseTTS
from app.config import settings

class EdgeTTSProvider(BaseTTS):
    """
    Microsoft Edge Neural Speech Provider (100% Free, high quality neural voices).
    Synthesizes streaming audio frames directly from text stream.
    """
    def __init__(self, voice: Optional[str] = None):
        self.default_voice = voice or settings.DEFAULT_TTS_VOICE

    async def synthesize_text(self, text: str, voice: Optional[str] = None) -> bytes:
        if not text.strip():
            return b""
        target_voice = voice or self.default_voice
        communicate = edge_tts.Communicate(text, target_voice, rate=settings.DEFAULT_TTS_RATE)
        
        audio_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])
        return audio_buffer.getvalue()

    async def synthesize_stream(
        self,
        text_stream: AsyncGenerator[str, None],
        voice: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Buffer text sentences/chunks from the LLM and stream synthesized audio chunks.
        """
        target_voice = voice or self.default_voice
        buffer = ""
        
        async for text_chunk in text_stream:
            if not text_chunk:
                continue
            buffer += text_chunk
            
            # Synthesize on sentence boundaries or punctuation to achieve low time-to-first-audio chunk
            if any(p in buffer for p in [".", "!", "?", "\n", ",", ";"]):
                # Extract up to punctuation mark
                speech_text = buffer.strip()
                buffer = ""
                if speech_text:
                    communicate = edge_tts.Communicate(speech_text, target_voice, rate=settings.DEFAULT_TTS_RATE)
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            yield chunk["data"]

        # Flush any remaining text in buffer
        if buffer.strip():
            communicate = edge_tts.Communicate(buffer.strip(), target_voice, rate=settings.DEFAULT_TTS_RATE)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]


class MockTTSProvider(BaseTTS):
    """Offline Mock TTS Provider for latency benchmark testing."""
    async def synthesize_text(self, text: str, voice: Optional[str] = None) -> bytes:
        # 1 second of dummy silent PCM audio
        return b"\x00" * 32000

    async def synthesize_stream(
        self,
        text_stream: AsyncGenerator[str, None],
        voice: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        async for text_chunk in text_stream:
            if text_chunk:
                yield b"\x00" * 3200
