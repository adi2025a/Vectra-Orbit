import numpy as np
import pytest
from app.providers.vad.target_speaker_vad import TargetSpeakerVAD
from app.providers.vad.energy_vad import EnergyVAD
from app.providers.factory import ProviderFactory


def generate_sine_pcm(freq: float = 440.0, duration_sec: float = 0.2, sample_rate: int = 16000) -> bytes:
    """Generate PCM 16-bit audio bytes for a given frequency."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    return samples.tobytes()


def test_factory_instantiation():
    vad = ProviderFactory.get_vad("target_speaker")
    assert isinstance(vad, TargetSpeakerVAD)


def test_target_speaker_auto_enrollment():
    # Use EnergyVAD as base for quick deterministic test
    base_vad = EnergyVAD(threshold_db=-60.0)
    ts_vad = TargetSpeakerVAD(base_vad=base_vad, enroll_seconds=0.4, threshold=0.5)

    assert not ts_vad.is_enrolled
    assert ts_vad.target_embedding is None

    # Feed 440 Hz audio chunks to complete enrollment (~0.4s = 2 chunks of 0.2s)
    chunk1 = generate_sine_pcm(freq=440.0, duration_sec=0.2)
    chunk2 = generate_sine_pcm(freq=440.0, duration_sec=0.25)

    res1 = ts_vad.process_chunk(chunk1)
    assert res1 is True
    assert not ts_vad.is_enrolled

    res2 = ts_vad.process_chunk(chunk2)
    assert res2 is True
    assert ts_vad.is_enrolled
    assert ts_vad.target_embedding is not None
    assert len(ts_vad.target_embedding) == 64


def test_target_speaker_matching_and_filtering():
    base_vad = EnergyVAD(threshold_db=-60.0)
    ts_vad = TargetSpeakerVAD(base_vad=base_vad, enroll_seconds=0.3, threshold=0.85)

    # Enroll target speaker voice (440 Hz tone)
    target_pcm = generate_sine_pcm(freq=440.0, duration_sec=0.4)
    ts_vad.process_chunk(target_pcm)
    assert ts_vad.is_enrolled

    # Matching voice chunk (440 Hz tone) -> should pass (True)
    match_chunk = generate_sine_pcm(freq=440.0, duration_sec=0.4)
    assert ts_vad.process_chunk(match_chunk) is True

    # Non-target background voice / TV speech chunk (2500 Hz tone) -> should be filtered (False)
    bg_tv_chunk = generate_sine_pcm(freq=2500.0, duration_sec=0.4)
    assert ts_vad.process_chunk(bg_tv_chunk) is False


def test_manual_embedding_override_and_reset():
    base_vad = EnergyVAD(threshold_db=-60.0)
    ts_vad = TargetSpeakerVAD(base_vad=base_vad)

    custom_emb = np.random.randn(64).astype(np.float32)
    ts_vad.set_target_embedding(custom_emb)
    assert ts_vad.is_enrolled
    assert ts_vad.target_embedding is not None

    ts_vad.reset()
    # Regular reset keeps locked speaker profile
    assert ts_vad.is_enrolled

    ts_vad.full_reset()
    # Full reset clears locked speaker profile for new call session
    assert not ts_vad.is_enrolled
    assert ts_vad.target_embedding is None
