import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class TurnMetric(BaseModel):
    turn_id: int
    user_speech_duration_ms: float = 0.0
    vad_latency_ms: float = 0.0
    stt_latency_ms: float = 0.0
    llm_ttft_ms: float = 0.0                 # Time To First Token
    llm_total_latency_ms: float = 0.0
    tts_first_chunk_ms: float = 0.0           # Time To First Audio Chunk
    tts_total_latency_ms: float = 0.0
    e2e_voice_latency_ms: float = 0.0        # User Speech End -> 1st Audio Chunk
    max_latency_module: str = "N/A"           # Which component was the bottleneck

    def calculate_bottleneck(self) -> str:
        latencies = {
            "STT": self.stt_latency_ms,
            "LLM (TTFT)": self.llm_ttft_ms,
            "TTS (1st Chunk)": self.tts_first_chunk_ms,
            "VAD": self.vad_latency_ms,
        }
        max_mod = max(latencies, key=latencies.get)
        self.max_latency_module = f"{max_mod} ({latencies[max_mod]:.1f}ms)"
        return self.max_latency_module

class LatencyTracker:
    """Helper timer context manager for tracking precise component latencies."""
    def __init__(self):
        self._start_times: Dict[str, float] = {}

    def start(self, key: str):
        self._start_times[key] = time.perf_counter()

    def stop(self, key: str) -> float:
        if key in self._start_times:
            elapsed = (time.perf_counter() - self._start_times[key]) * 1000.0
            return elapsed
        return 0.0

class CallMetricsSummary(BaseModel):
    session_id: str
    total_turns: int = 0
    avg_e2e_latency_ms: float = 0.0
    avg_stt_latency_ms: float = 0.0
    avg_llm_ttft_ms: float = 0.0
    avg_tts_first_chunk_ms: float = 0.0
    max_bottleneck_summary: str = ""
    turns: List[TurnMetric] = []

    def aggregate(self):
        if not self.turns:
            return
        self.total_turns = len(self.turns)
        self.avg_e2e_latency_ms = sum(t.e2e_voice_latency_ms for t in self.turns) / self.total_turns
        self.avg_stt_latency_ms = sum(t.stt_latency_ms for t in self.turns) / self.total_turns
        self.avg_llm_ttft_ms = sum(t.llm_ttft_ms for t in self.turns) / self.total_turns
        self.avg_tts_first_chunk_ms = sum(t.tts_first_chunk_ms for t in self.turns) / self.total_turns
