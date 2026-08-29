import pytest
from app.core.metrics import TurnMetric, LatencyTracker, CallMetricsSummary
from app.providers.factory import ProviderFactory

def test_latency_tracker():
    tracker = LatencyTracker()
    tracker.start("stt")
    import time
    time.sleep(0.01)  # 10ms
    elapsed = tracker.stop("stt")
    assert elapsed >= 8.0  # At least ~10ms

def test_turn_metric_bottleneck():
    turn = TurnMetric(
        turn_id=1,
        vad_latency_ms=5.0,
        stt_latency_ms=180.0,
        llm_ttft_ms=75.0,
        tts_first_chunk_ms=40.0,
        e2e_voice_latency_ms=300.0
    )
    bottleneck = turn.calculate_bottleneck()
    assert "STT" in bottleneck
    assert "180.0ms" in bottleneck

def test_provider_factory_mock():
    stt = ProviderFactory.get_stt("mock")
    llm = ProviderFactory.get_llm("mock")
    gemini = ProviderFactory.get_llm("gemini")
    tts = ProviderFactory.get_tts("mock")
    vad = ProviderFactory.get_vad("energy")

    assert stt is not None
    assert llm is not None
    assert gemini is not None
    assert tts is not None
    assert vad is not None
