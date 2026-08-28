"""Interfaces / Contracts for modular cascaded voice calling components."""
from .vad_interface import BaseVAD
from .stt_interface import BaseSTT
from .tts_interface import BaseTTS
from .llm_interface import BaseLLM, LLMChunk, FunctionCall
from .telephony_interface import BaseTelephonyAdapter

__all__ = [
    "BaseVAD",
    "BaseSTT",
    "BaseTTS",
    "BaseLLM",
    "LLMChunk",
    "FunctionCall",
    "BaseTelephonyAdapter",
]
