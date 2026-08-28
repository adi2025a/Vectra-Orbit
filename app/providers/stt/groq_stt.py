import io
import httpx
import wave
from typing import Optional
from app.interfaces.stt_interface import BaseSTT
from app.config import settings

class GroqSTT(BaseSTT):
    """
    Groq Cloud Whisper STT provider (Free Cloud Tier).
    Extremely fast (~150ms transcription latency).
    """
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model or settings.GROQ_STT_MODEL
        self.url = "https://api.groq.com/openai/v1/audio/transcriptions"

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000, language: Optional[str] = "en") -> str:
        if not audio_bytes or len(audio_bytes) < 1000:
            return ""

        if not self.api_key:
            # Fall back to clear warning / mock
            return "[Error: GROQ_API_KEY missing. Please set your free GROQ_API_KEY in .env or switch STT_PROVIDER to mock]"

        # Wrap raw PCM 16-bit mono audio bytes into a valid WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)
        
        wav_buffer.seek(0)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        files = {
            "file": ("audio.wav", wav_buffer.read(), "audio/wav")
        }
        data = {
            "model": self.model,
            "language": language or "en",
            "response_format": "json"
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(self.url, headers=headers, files=files, data=data)
            if response.status_code == 200:
                result = response.json()
                return result.get("text", "").strip()
            else:
                return f"[STT Error {response.status_code}: {response.text}]"


class MockSTT(BaseSTT):
    """Offline Mock STT Provider for testing without API keys."""
    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000, language: Optional[str] = "en") -> str:
        if len(audio_bytes) < 1000:
            return ""
        return "Hello! I am testing the AI calling platform. Can you forward me to customer support?"
