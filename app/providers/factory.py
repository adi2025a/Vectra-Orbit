from app.config import settings
from app.interfaces.vad_interface import BaseVAD
from app.interfaces.stt_interface import BaseSTT
from app.interfaces.llm_interface import BaseLLM
from app.interfaces.tts_interface import BaseTTS
from app.interfaces.telephony_interface import BaseTelephonyAdapter

from app.providers.vad.silero_vad import SileroVAD
from app.providers.vad.energy_vad import EnergyVAD
from app.providers.vad.target_speaker_vad import TargetSpeakerVAD
from app.providers.stt.groq_stt import GroqSTT, MockSTT
from app.providers.llm.groq_llm import GroqLLM, OllamaLLM, MockLLM
from app.providers.llm.gemini_llm import GeminiLLM
from app.providers.tts.edge_tts_provider import EdgeTTSProvider, MockTTSProvider
from app.providers.telephony.web_telephony import WebTelephonyAdapter, TwilioTelephonyAdapter
from fastapi import WebSocket

class ProviderFactory:
    """Central Factory for initializing configured VAD, STT, LLM, TTS, and Telephony modules."""

    @staticmethod
    def get_vad(provider_name: str | None = None) -> BaseVAD:
        name = (provider_name or settings.VAD_PROVIDER).lower()
        if name == "target_speaker":
            return TargetSpeakerVAD(base_vad=SileroVAD())
        elif name == "silero":
            return SileroVAD()
        elif name == "energy":
            return EnergyVAD()
        return EnergyVAD()

    @staticmethod
    def get_stt(provider_name: str | None = None) -> BaseSTT:
        name = (provider_name or settings.STT_PROVIDER).lower()
        if name == "groq":
            return GroqSTT()
        elif name == "mock":
            return MockSTT()
        return GroqSTT()

    @staticmethod
    def get_llm(provider_name: str | None = None) -> BaseLLM:
        name = (provider_name or settings.LLM_PROVIDER).lower()
        if name == "groq":
            return GroqLLM()
        elif name == "gemini":
            return GeminiLLM()
        elif name == "ollama":
            return OllamaLLM()
        elif name == "mock":
            return MockLLM()
        return GroqLLM()

    @staticmethod
    def get_tts(provider_name: str | None = None) -> BaseTTS:
        name = (provider_name or settings.TTS_PROVIDER).lower()
        if name == "edge":
            return EdgeTTSProvider()
        elif name == "mock":
            return MockTTSProvider()
        return EdgeTTSProvider()

    @staticmethod
    def get_telephony(websocket: WebSocket, provider_name: str | None = None) -> BaseTelephonyAdapter:
        name = (provider_name or settings.TELEPHONY_PROVIDER).lower()
        if name == "twilio":
            return TwilioTelephonyAdapter(websocket)
        return WebTelephonyAdapter(websocket)
