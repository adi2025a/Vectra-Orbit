import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "Vectra-Orbit AI Calling Platform"
    DEBUG: bool = True

    # Database
    # Default to zero-setup local SQLite via aiosqlite, can be overridden to PostgreSQL via env (postgresql+asyncpg://...)
    DATABASE_URL: str = "sqlite+aiosqlite:///./vectra_orbit.db"

    # Active Providers (Easily switchable via ENV or API)
    VAD_PROVIDER: str = "silero"          # "silero", "energy", "target_speaker"
    TARGET_SPEAKER_THRESHOLD: float = 0.60
    TARGET_SPEAKER_ENROLL_SECONDS: float = 1.0
    STT_PROVIDER: str = "groq"            # "groq", "faster_whisper", "mock"
    LLM_PROVIDER: str = "gemini"            # "groq", "gemini", "ollama", "mock"
    TTS_PROVIDER: str = "edge"            # "edge", "mock"
    TELEPHONY_PROVIDER: str = "web"       # "web", "twilio"

    # API Keys & Endpoints for Zero-Cost/Free Tiers
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    GROQ_STT_MODEL: str = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")

    # Default Voice for Edge TTS (Free Microsoft Neural Voices)
    DEFAULT_TTS_VOICE: str = "en-US-AvaNeural"
    DEFAULT_TTS_RATE: str = "+0%"

    # Audio Pipeline & VAD Settings
    SAMPLE_RATE: int = 16000             # 16kHz audio standard for VAD & STT
    CHANNELS: int = 1                     # Mono audio
    CHUNK_SIZE_MS: int = 20               # 20ms audio chunks from WebSocket stream
    VAD_BARGEIN_SPEECH_CHUNKS: int = 6    # Continuous speech chunks (~120ms) required to trigger barge-in
    VAD_SILENCE_CHUNKS: int = 12          # Continuous silence chunks (~240ms) to end user turn
    ENABLE_BARGEIN: bool = False           # Toggle barge-in interruption on/off
    TTS_MIN_CHUNK_WORDS: int = 4          # Min words to trigger early streaming TTS synthesis

    # Default Handoff/Forwarding Phone Number
    DEFAULT_HUMAN_AGENT_PHONE: str = os.getenv("DEFAULT_HUMAN_AGENT_PHONE", "+18005550199")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
