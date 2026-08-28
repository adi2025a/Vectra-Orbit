import numpy as np

def resample_pcm16(audio_bytes: bytes, orig_sr: int, target_sr: int = 16000) -> bytes:
    """Resample 16-bit signed PCM audio bytes to target sample rate."""
    if orig_sr == target_sr or not audio_bytes:
        return audio_bytes

    samples = np.frombuffer(audio_bytes, dtype=np.int16)
    duration = len(samples) / orig_sr
    target_length = int(duration * target_sr)
    
    if target_length <= 0:
        return b""

    old_indices = np.linspace(0, len(samples) - 1, len(samples))
    new_indices = np.linspace(0, len(samples) - 1, target_length)
    resampled_samples = np.interp(new_indices, old_indices, samples).astype(np.int16)
    return resampled_samples.tobytes()

def pcm2float(audio_bytes: bytes) -> np.ndarray:
    """Convert PCM 16-bit int bytes to float32 normalized array [-1.0, 1.0]."""
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

def float2pcm(float_samples: np.ndarray) -> bytes:
    """Convert float32 normalized array [-1.0, 1.0] back to PCM 16-bit int bytes."""
    int_samples = (np.clip(float_samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    return int_samples.tobytes()
